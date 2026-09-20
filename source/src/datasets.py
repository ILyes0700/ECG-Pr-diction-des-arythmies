"""Chargement MIT-BIH / INCART / SVDB et construction des ensembles.

Logique d'evaluation (verrouillee et verifiee par des assertions) :

    DS1  ->  train (17 patients)  +  validation (5 patients)   [aucun patient commun]
    DS2  ->  test  (22 patients)                               [jamais vu a l'entrainement]

    Adaptation patient (optionnelle, protocole AAMI) :
      les battements des ADAPT_SECONDS premieres secondes de chaque enregistrement DS2
      servent au reglage fin ; ils sont RETIRES du test. Le test reste donc
      strictement disjoint des donnees vues par le modele.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from . import config as C
from .preprocessing import (preprocess_signal, resample_to, zscore_beat, rr_features,
                            stack_context_by_record, RRScaler)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_records(data_dir, records, fs_target=C.FS, cache_name=None, verbose=True):
    """Retourne X (n, L), RR (n, 6), y (n,), groups (n,), t (n,) en secondes, durees."""
    import wfdb
    data_dir = Path(data_dir)
    cache = None
    if cache_name:
        C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = C.CACHE_DIR / f"{cache_name}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if verbose:
                print(f"  cache -> {cache.name}  ({len(z['y']):,} battements)")
            return (z["X"], z["RR"], z["y"], z["g"], z["t"],
                    dict(zip(z["dur_keys"].tolist(), z["dur_vals"].tolist())))

    X, RR, Y, G, T = [], [], [], [], []
    durations = {}
    for rec in records:
        path = data_dir / str(rec)
        try:
            record = wfdb.rdrecord(str(path))
            ann = wfdb.rdann(str(path), "atr")
        except Exception as exc:                      # enregistrement absent -> ignore
            if verbose:
                print(f"  [skip] {rec}: {exc}")
            continue
        fs = float(record.fs)
        sig = preprocess_signal(record.p_signal[:, 0], fs, C.BP_LOW, C.BP_HIGH)
        durations[str(rec)] = len(sig) / fs

        keep = [i for i, s in enumerate(ann.symbol) if s in C.AAMI_MAP]
        if not keep:
            continue
        peaks = ann.sample[np.asarray(keep)]
        labels = [C.AAMI_MAP[ann.symbol[i]] for i in keep]
        rrf = rr_features(peaks, fs)

        pre = int(round(C.PRE * fs / C.FS))           # fenetre exprimee en secondes
        post = int(round(C.POST * fs / C.FS))
        n_ok = 0
        for k, (pk, lab) in enumerate(zip(peaks, labels)):
            a, b = int(pk) - pre, int(pk) + post
            if a < 0 or b > len(sig):
                continue
            beat = resample_to(sig[a:b], C.RESAMPLE_LEN)
            X.append(zscore_beat(beat))
            RR.append(rrf[k])
            Y.append(C.CLASS_IDX[lab])
            G.append(str(rec))
            T.append(float(pk) / fs)                  # instant du battement (s)
            n_ok += 1
        if verbose:
            print(f"  {rec}: {n_ok:>6,} battements | {durations[str(rec)]/60:.1f} min")

    X = np.asarray(X, np.float32); RR = np.asarray(RR, np.float32)
    Y = np.asarray(Y, np.int64); G = np.asarray(G); T = np.asarray(T, np.float64)
    if cache is not None:
        np.savez_compressed(cache, X=X, RR=RR, y=Y, g=G, t=T,
                            dur_keys=np.array(list(durations.keys())),
                            dur_vals=np.array(list(durations.values())))
    return X, RR, Y, G, T, durations


# --------------------------------------------------------------------------- #
# Construction des ensembles
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    Xtr: np.ndarray; Rtr: np.ndarray; ytr: np.ndarray; gtr: np.ndarray
    Xva: np.ndarray; Rva: np.ndarray; yva: np.ndarray; gva: np.ndarray
    Xte: np.ndarray; Rte: np.ndarray; yte: np.ndarray; gte: np.ndarray
    Xad: np.ndarray; Rad: np.ndarray; yad: np.ndarray; gad: np.ndarray
    rr_scaler: RRScaler
    durations: dict = field(default_factory=dict)
    class_weights: dict = field(default_factory=dict)
    # Instants des battements (s) : necessaires a la regle de declenchement d'alarme.
    tte: np.ndarray = None
    tva: np.ndarray = None

    def summary(self):
        return {"train": len(self.ytr), "validation": len(self.yva), "test": len(self.yte),
                "adaptation": len(self.yad),
                "patients_train": len(np.unique(self.gtr)),
                "patients_val": len(np.unique(self.gva)),
                "patients_test": len(np.unique(self.gte))}


def compute_class_weights(counts):
    """Poids de classe en puissance `CLASS_WEIGHT_POWER`, normalises a moyenne 1 et plafonnes.

    L'inverse pur de la frequence donne un rapport N:F de 1:90 sur MIT-BIH, ce qui sature
    la perte focale et fait predire les classes rares partout (Accuracy 0.10, VPP_S 0.06).
    """
    counts = np.asarray(counts, dtype=float)
    present = counts > 0
    p = float(getattr(C, "CLASS_WEIGHT_POWER", 0.5))
    cap = float(getattr(C, "MAX_CLASS_WEIGHT", 10.0))
    w = np.zeros_like(counts)
    w[present] = (counts[present].sum() / counts[present]) ** p
    w[present] /= w[present].mean()
    return {i: float(v) for i, v in enumerate(np.clip(w, 0.0, cap))}


def build_split(X, RR, y, g, t, durations, use_adaptation=True,
                adapt_seconds=C.ADAPT_SECONDS, verbose=True):
    train_recs = [r for r in C.DS1 if r not in C.VAL_RECORDS]
    val_recs = list(C.VAL_RECORDS)
    test_recs = list(C.DS2)

    # --- garde-fous sur le protocole lui-meme --------------------------------
    assert not (set(train_recs) & set(val_recs)), "train et validation partagent un patient"
    assert not (set(C.DS1) & set(C.DS2)), "DS1 et DS2 partagent un patient"
    assert not (set(C.DS1 + C.DS2) & set(C.PACED_RECORDS)), "un enregistrement stimule est inclus"

    m_tr = np.isin(g, train_recs)
    m_va = np.isin(g, val_recs)
    m_te_all = np.isin(g, test_recs)

    # --- decoupe temporelle du test pour l'adaptation patient ----------------
    if use_adaptation:
        m_ad = m_te_all & (t < adapt_seconds)          # 5 premieres minutes = connues
        m_te = m_te_all & (t >= adapt_seconds)         # le reste = test reel
    else:
        m_ad = np.zeros_like(m_te_all)
        m_te = m_te_all
    assert not np.any(m_ad & m_te), "chevauchement entre adaptation et test"

    # --- contexte multi-battements, calcule par patient ----------------------
    Xc = stack_context_by_record(X, g)

    # --- RR : scaler ajuste sur le TRAIN uniquement --------------------------
    scaler = RRScaler().fit(RR[m_tr])

    def take(mask):
        return (Xc[mask], scaler.transform(RR[mask]), y[mask], g[mask])

    Xtr, Rtr, ytr, gtr = take(m_tr)
    Xva, Rva, yva, gva = take(m_va)
    Xte, Rte, yte, gte = take(m_te)
    Xad, Rad, yad, gad = take(m_ad)

    # --- verifications anti-fuite --------------------------------------------
    assert not (set(np.unique(gtr)) & set(np.unique(gva)))
    assert not (set(np.unique(gtr)) & set(np.unique(gte)))
    assert not (set(np.unique(gva)) & set(np.unique(gte)))

    class_weights = compute_class_weights(np.bincount(ytr, minlength=C.N_CLASSES))

    split = Split(Xtr, Rtr, ytr, gtr, Xva, Rva, yva, gva, Xte, Rte, yte, gte,
                  Xad, Rad, yad, gad, scaler, durations, class_weights,
                  tte=np.asarray(t)[m_te], tva=np.asarray(t)[m_va])
    if verbose:
        print("Composition des ensembles :", split.summary())
        print("Poids de classes (train) :",
              {C.CLASS_NAMES[k]: round(v, 3) for k, v in class_weights.items()})
        print("Aucun patient partage entre train / validation / test. Verifie.")
    return split


def test_durations_after_adaptation(durations, use_adaptation=True,
                                    adapt_seconds=C.ADAPT_SECONDS):
    """Duree reellement evaluee par enregistrement (pour les alarmes par heure)."""
    out = {}
    for rec in C.DS2:
        d = durations.get(str(rec))
        if d is None:
            continue
        out[str(rec)] = max(d - adapt_seconds, 0.0) if use_adaptation else d
    return out
