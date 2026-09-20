"""Controle qualite du signal et nettoyage des battements.

Regle scientifique importante : on ne supprime JAMAIS de battements du jeu de test.
Retirer des battements difficiles du test gonflerait artificiellement les scores et serait
rejete par un relecteur. Le nettoyage agit donc :

  - sur le TRAIN : les battements inexploitables sont retires (ils n'apprennent rien de bon) ;
  - sur le TEST  : ils sont seulement COMPTES et rapportes (Table de qualite du signal).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Indices de qualite, calcules battement par battement
# --------------------------------------------------------------------------- #
def beat_quality(beats):
    """Retourne un DataFrame d'indicateurs (une ligne par battement).

    - nan_inf        : valeurs non finies
    - flat           : segment plat (electrode debranchee)
    - saturated      : ecretage de l'amplificateur
    - amplitude      : amplitude crete-a-crete (apres z-score)
    - kurtosis (kSQI): un ECG propre a un kurtosis eleve (QRS pointu) ; un bruit
                       gaussien tend vers 3. kSQI < 2 signale un battement noye dans le bruit.
    - hf_ratio       : energie haute frequence / energie totale (bruit EMG)
    """
    X = np.asarray(beats, dtype=np.float64)
    if X.ndim == 3:                      # (n, L, canaux) -> canal central
        X = X[:, :, X.shape[2] // 2]

    finite = np.isfinite(X).all(axis=1)
    Xs = np.where(np.isfinite(X), X, 0.0)

    std = Xs.std(axis=1)
    ptp = Xs.max(axis=1) - Xs.min(axis=1)

    mu = Xs.mean(axis=1, keepdims=True)
    sd = Xs.std(axis=1, keepdims=True) + 1e-12
    z = (Xs - mu) / sd
    kurt = (z ** 4).mean(axis=1)

    diff = np.diff(Xs, axis=1)
    hf = (diff ** 2).sum(axis=1) / ((Xs ** 2).sum(axis=1) + 1e-12)

    # ecretage : beaucoup d'echantillons colles a la valeur extreme
    top = np.isclose(Xs, Xs.max(axis=1, keepdims=True), rtol=0, atol=1e-6).sum(axis=1)
    bot = np.isclose(Xs, Xs.min(axis=1, keepdims=True), rtol=0, atol=1e-6).sum(axis=1)
    saturated = (top + bot) > 0.10 * Xs.shape[1]

    return pd.DataFrame({
        "nan_inf": ~finite,
        "flat": std < 1e-3,
        "saturated": saturated,
        "amplitude": ptp,
        "kSQI": kurt,
        "hf_ratio": hf,
    })


def quality_mask(beats, kurtosis_min=2.0, hf_max=0.6, amplitude_min=0.05):
    """Masque booleen : True = battement exploitable."""
    q = beat_quality(beats)
    keep = (~q["nan_inf"]) & (~q["flat"]) & (~q["saturated"]) \
        & (q["kSQI"] >= kurtosis_min) & (q["hf_ratio"] <= hf_max) \
        & (q["amplitude"] >= amplitude_min)
    return np.array(keep.to_numpy(), dtype=bool), q


def quality_report(beats, labels, class_names, name="ensemble"):
    """Table de qualite : combien de battements douteux, et dans quelle classe."""
    keep, q = quality_mask(beats)
    rows = []
    for i, c in enumerate(class_names):
        m = np.asarray(labels) == i
        if not m.any():
            continue
        rows.append({
            "Ensemble": name, "Classe": c, "Battements": int(m.sum()),
            "NaN/Inf": int(q["nan_inf"][m].sum()),
            "Plats": int(q["flat"][m].sum()),
            "Satures": int(q["saturated"][m].sum()),
            "Bruit_HF": int((q["hf_ratio"][m] > 0.6).sum()),
            "kSQI_faible": int((q["kSQI"][m] < 2.0).sum()),
            "Rejetes": int((~keep[m]).sum()),
            "Taux_rejet_%": round(100 * float((~keep[m]).mean()), 2),
        })
    total = {"Ensemble": name, "Classe": "TOTAL", "Battements": int(len(labels)),
             "NaN/Inf": int(q["nan_inf"].sum()), "Plats": int(q["flat"].sum()),
             "Satures": int(q["saturated"].sum()), "Bruit_HF": int((q["hf_ratio"] > 0.6).sum()),
             "kSQI_faible": int((q["kSQI"] < 2.0).sum()), "Rejetes": int((~keep).sum()),
             "Taux_rejet_%": round(100 * float((~keep).mean()), 2)}
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([total])], ignore_index=True), keep


def clean_training_set(X, R, y, g, verbose=True, min_per_class=50):
    """Retire les battements inexploitables du TRAIN, en protegeant les classes rares."""
    keep, _ = quality_mask(X)
    keep = np.array(keep, dtype=bool, copy=True)
    y = np.asarray(y)
    # securite : ne jamais vider une classe rare (F, S) a cause du controle qualite
    for c in np.unique(y):
        m = y == c
        if keep[m].sum() < min(min_per_class, m.sum()):
            keep[m] = True
            if verbose:
                print(f"  classe {c} : rejet annule (classe trop rare pour etre filtree)")
    if verbose:
        print(f"  train nettoye : {int((~keep).sum()):,} battements retires "
              f"sur {len(keep):,} ({100*(~keep).mean():.2f} %)")
    return X[keep], R[keep], y[keep], np.asarray(g)[keep], keep
