"""Incertitude predictive et option de rejet (renvoi au cardiologue)."""
from __future__ import annotations
import numpy as np


def predictive_entropy(proba):
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-12, 1.0)
    return (-(p * np.log(p)).sum(axis=1)).astype(np.float32)


def mc_dropout_predict(model, inputs, passes=30, batch_size=512):
    """Moyenne et incertitude par Monte-Carlo dropout (dropout actif a l'inference)."""
    preds = []
    for _ in range(passes):
        preds.append(np.asarray(model(inputs, training=True)))
    preds = np.stack(preds)                       # (passes, n, classes)
    mean = preds.mean(0)
    epistemic = preds.var(0).sum(1)               # desaccord entre passes
    total = predictive_entropy(mean)
    return mean.astype(np.float32), epistemic.astype(np.float32), total


def deferral_for_target_accuracy(y_true, proba, uncertainty, target_accuracy=0.99):
    """Taux de cas a renvoyer a un expert pour atteindre la fiabilite visee."""
    y = np.asarray(y_true, int)
    correct = (proba.argmax(1) == y).astype(float)
    order = np.argsort(np.asarray(uncertainty))   # du plus sûr au plus incertain
    c = correct[order]
    running = np.cumsum(c) / np.arange(1, len(c) + 1)
    ok = np.flatnonzero(running >= target_accuracy)
    if len(ok) == 0:
        return {"couverture": 0.0, "taux_renvoi": 1.0, "exactitude_atteinte": float(c.mean())}
    k = int(ok[-1]) + 1
    return {"couverture": k / len(c), "taux_renvoi": 1.0 - k / len(c),
            "exactitude_atteinte": float(running[k - 1])}
