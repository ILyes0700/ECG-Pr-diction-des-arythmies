"""Pretraitement du signal, segmentation contextuelle, descripteurs RR, augmentation.

Regle d'or respectee partout : aucune statistique n'est calculee sur le test.
- normalisation morphologique : z-score PAR BATTEMENT (aucune statistique globale) ;
- descripteurs RR : standardises avec un scaler ajuste sur le TRAIN uniquement ;
- augmentation : appliquee au TRAIN uniquement.
"""
from __future__ import annotations
import numpy as np

try:                       # scipy est requis pour le filtrage, pas pour les tests unitaires
    import scipy.signal as sps
except Exception:          # pragma: no cover
    sps = None


# --------------------------------------------------------------------------- #
# Filtrage
# --------------------------------------------------------------------------- #
def bandpass(sig, fs, low=0.5, high=40.0, order=4):
    b, a = sps.butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return sps.filtfilt(b, a, sig)


def remove_baseline(sig, fs):
    """Deux filtres medians (200 ms puis 600 ms) : methode de reference en ECG."""
    k1 = int(0.2 * fs) | 1
    k2 = int(0.6 * fs) | 1
    return sig - sps.medfilt(sps.medfilt(sig, k1), k2)


def preprocess_signal(sig, fs, low=0.5, high=40.0):
    sig = np.asarray(sig, dtype=np.float64)
    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return bandpass(remove_baseline(sig, fs), fs, low, high)


def resample_to(sig, length):
    return sps.resample(sig, length)


# --------------------------------------------------------------------------- #
# Normalisation et descripteurs
# --------------------------------------------------------------------------- #
def zscore_beat(beat):
    beat = np.asarray(beat, dtype=np.float32)
    return (beat - beat.mean()) / (beat.std() + 1e-8)


def rr_features(r_samples, fs, local_window=10):
    """6 descripteurs rythmiques par battement.

    pre_rr, post_rr, rr_local(10), pre/local, pre/global, post/pre.
    La prematurite (pre/local < 1) est le marqueur cle de la classe S.
    """
    r = np.asarray(r_samples, dtype=np.float64) / float(fs)
    if len(r) < 2:
        return np.zeros((len(r), 6), dtype=np.float32)
    rr = np.diff(r)
    pre = np.concatenate([[rr[0]], rr])                  # RR precedent
    post = np.concatenate([rr, [rr[-1]]])                # RR suivant
    kernel = np.ones(local_window) / local_window
    padded = np.concatenate([np.full(local_window - 1, pre[0]), pre])
    local = np.convolve(padded, kernel, mode="valid")    # moyenne glissante causale
    global_rr = np.median(pre)
    feats = np.stack([pre, post, local,
                      pre / (local + 1e-8),
                      pre / (global_rr + 1e-8),
                      post / (pre + 1e-8)], axis=1)
    return np.clip(feats, -10, 10).astype(np.float32)


# --------------------------------------------------------------------------- #
# Segmentation contextuelle : battement precedent | courant | suivant
# --------------------------------------------------------------------------- #
def stack_context(beats):
    """(n, L) -> (n, L, 3). Les bords sont repliques (pas de fuite temporelle)."""
    beats = np.asarray(beats, dtype=np.float32)
    prev = np.vstack([beats[:1], beats[:-1]])
    nxt = np.vstack([beats[1:], beats[-1:]])
    return np.stack([prev, beats, nxt], axis=-1)


def stack_context_by_record(beats, groups):
    """Idem, mais le contexte ne franchit jamais la frontiere entre deux patients."""
    beats = np.asarray(beats, dtype=np.float32)
    groups = np.asarray(groups)
    out = np.zeros((len(beats), beats.shape[1], 3), dtype=np.float32)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = stack_context(beats[idx])
    return out


# --------------------------------------------------------------------------- #
# Augmentation (TRAIN uniquement)
# --------------------------------------------------------------------------- #
def augment_batch(X, rng, p=0.5, amp=(0.8, 1.2), noise_sd=0.05, shift_max=6, warp=0.1):
    """Bruit gaussien, variation de gain, decalage temporel, etirement leger.

    Simule les artefacts realistes (mouvement d'electrode, gain d'ampli, jitter du
    detecteur QRS) sans changer la classe du battement.
    """
    X = np.array(X, dtype=np.float32, copy=True)
    n, length, ch = X.shape
    sel = rng.random(n) < p
    if not sel.any():
        return X
    idx = np.flatnonzero(sel)
    X[idx] *= rng.uniform(amp[0], amp[1], size=(len(idx), 1, 1)).astype(np.float32)
    X[idx] += rng.normal(0, noise_sd, size=(len(idx), length, ch)).astype(np.float32)
    for i in idx:                                        # decalage circulaire faible
        s = int(rng.integers(-shift_max, shift_max + 1))
        if s:
            X[i] = np.roll(X[i], s, axis=0)
    if warp > 0:                                         # etirement/compression douce
        grid = np.arange(length)
        for i in idx:
            factor = 1.0 + float(rng.uniform(-warp, warp))
            src = np.clip(grid * factor, 0, length - 1)
            for c in range(ch):
                X[i, :, c] = np.interp(grid, src, X[i, :, c])
    return X


# --------------------------------------------------------------------------- #
# Standardisation des descripteurs RR (fit sur le train seul)
# --------------------------------------------------------------------------- #
class RRScaler:
    def fit(self, R):
        R = np.asarray(R, dtype=np.float64)
        self.mean_ = R.mean(0)
        self.std_ = R.std(0) + 1e-8
        return self

    def transform(self, R):
        return ((np.asarray(R, dtype=np.float64) - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, R):
        return self.fit(R).transform(R)
