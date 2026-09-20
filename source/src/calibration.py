"""Calibration probabiliste : temperature scaling ajuste sur la VALIDATION."""
from __future__ import annotations
import numpy as np


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=axis, keepdims=True)).astype(np.float32)


def negative_log_likelihood(proba, y):
    p = np.clip(proba[np.arange(len(y)), np.asarray(y, int)], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(logits_val, y_val, grid=None):
    """Recherche la temperature T minimisant la NLL sur la validation uniquement."""
    grid = np.linspace(0.5, 5.0, 181) if grid is None else np.asarray(grid)
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        nll = negative_log_likelihood(softmax(np.asarray(logits_val) / t), y_val)
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t, best_nll


def apply_temperature(logits, temperature):
    return softmax(np.asarray(logits) / float(temperature))


def fit_vector_scaling(logits_val, y_val, iters=800, lr=0.05, l2=1e-4):
    """Vector scaling (Guo et al. 2017) : une echelle et un biais PAR CLASSE.

    Le temperature scaling n'a qu'un parametre : il peut corriger une sur/sous-confiance
    globale mais pas un decalage de prior par classe. Or l'apprentissage pondere deplace
    justement le prior (la classe N est sous-estimee), d'ou un ECE eleve. Le vector
    scaling ajuste z_c -> w_c * z_c + b_c en minimisant la NLL sur la VALIDATION seule.
    Optimisation Adam en numpy : deterministe, quelques millisecondes.
    """
    z = np.asarray(logits_val, dtype=np.float64)
    y = np.asarray(y_val, dtype=int)
    n, k = z.shape
    onehot = np.zeros((n, k)); onehot[np.arange(n), y] = 1.0

    w = np.ones(k); b = np.zeros(k)
    mw = vw = np.zeros(k); mb = vb = np.zeros(k)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for it in range(1, iters + 1):
        p = softmax(w * z + b).astype(np.float64)
        dz = (p - onehot) / n
        gw = (dz * z).sum(0) + 2 * l2 * (w - 1.0)
        gb = dz.sum(0) + 2 * l2 * b
        mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw ** 2
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
        w -= lr * (mw / (1 - b1 ** it)) / (np.sqrt(vw / (1 - b2 ** it)) + eps)
        b -= lr * (mb / (1 - b1 ** it)) / (np.sqrt(vb / (1 - b2 ** it)) + eps)
    nll = negative_log_likelihood(softmax(w * z + b), y)
    return w, b, float(nll)


def apply_vector_scaling(logits, w, b):
    return softmax(np.asarray(logits, dtype=np.float64) * np.asarray(w) + np.asarray(b))


def proba_to_logits(proba, floor=1e-12):
    """Log-probabilites : permet de calibrer une moyenne d'ensemble sans acceder aux logits."""
    return np.log(np.clip(np.asarray(proba, dtype=np.float64), floor, 1.0))


def calibrate_ensemble(p_val, y_val, p_test):
    """Calibre une moyenne d'ensemble : temperature vs vector scaling, choix par NLL validation.

    Retourne (p_test_calibre, p_val_calibre, info) ou info documente la methode retenue.
    """
    zv, zt = proba_to_logits(p_val), proba_to_logits(p_test)
    T, nll_T = fit_temperature(zv, y_val)
    w, b, nll_V = fit_vector_scaling(zv, y_val)
    nll_id = negative_log_likelihood(softmax(zv), y_val)
    cands = {
        "aucune": (nll_id, softmax(zt), softmax(zv), {"methode": "aucune"}),
        "temperature": (nll_T, apply_temperature(zt, T), apply_temperature(zv, T),
                        {"methode": "temperature", "T": round(float(T), 4)}),
        "vector": (nll_V, apply_vector_scaling(zt, w, b), apply_vector_scaling(zv, w, b),
                   {"methode": "vector_scaling",
                    "w": [round(float(x), 4) for x in w],
                    "b": [round(float(x), 4) for x in b]}),
    }
    best = min(cands, key=lambda k: cands[k][0])
    nll, pt, pv, info = cands[best]
    info.update({"nll_validation": round(float(nll), 5),
                 "nll_sans_calibration": round(float(nll_id), 5),
                 "nll_temperature": round(float(nll_T), 5),
                 "nll_vector": round(float(nll_V), 5),
                 "temperature": round(float(T), 4)})
    return pt.astype(np.float32), pv.astype(np.float32), info
