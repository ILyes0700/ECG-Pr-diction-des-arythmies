"""Metriques diagnostiques et sanitaires (numpy pur : testable sans TensorFlow).

Toutes les fonctions prennent des etiquettes entieres et, si besoin, les probabilites.
Aucune ne modifie ses entrees. Aucune ne depend de scikit-learn : les formules sont
explicites, ce qui facilite la relecture par un reviewer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Briques de base
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true, int); y_pred = np.asarray(y_pred, int)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    # Robustesse : un cache genere avec un nombre de classes different ne doit pas
    # faire planter l'evaluation ; les etiquettes hors domaine sont ignorees.
    ok = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    np.add.at(cm, (y_true[ok], y_pred[ok]), 1)
    return cm


def binary_counts(cm, i):
    """TP, FP, FN, TN de la classe i en un-contre-tous."""
    tp = cm[i, i]
    fn = cm[i].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - tp - fn - fp
    return int(tp), int(fp), int(fn), int(tn)


def _div(a, b):
    return float(a) / float(b) if b else np.nan


# --------------------------------------------------------------------------- #
# Metriques diagnostiques par classe (Table IV de l'article)
# --------------------------------------------------------------------------- #
def clinical_table(y_true, y_pred, class_names, prevalence_grid=(0.01, 0.05, 0.10, 0.20)):
    """Se, Sp, VPP, VPN, F1, LR+, LR-, DOR, Youden J, MCC, NND, VPP ajustes."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, n)
    rows = []
    for i, name in enumerate(class_names):
        tp, fp, fn, tn = binary_counts(cm, i)
        se = _div(tp, tp + fn)                      # sensibilite / rappel
        sp = _div(tn, tn + fp)                      # specificite
        ppv = _div(tp, tp + fp)                     # valeur predictive positive
        npv = _div(tn, tn + fn)                     # valeur predictive negative
        f1 = _div(2 * tp, 2 * tp + fp + fn)
        fpr = _div(fp, fp + tn)
        fnr = _div(fn, fn + tp)
        lr_pos = _div(se, 1 - sp) if sp is not np.nan else np.nan
        lr_neg = _div(1 - se, sp) if sp else np.nan
        dor = _div(lr_pos, lr_neg) if (lr_neg and not np.isnan(lr_neg)) else np.nan
        youden = se + sp - 1 if not (np.isnan(se) or np.isnan(sp)) else np.nan
        nnd = _div(1.0, youden) if youden and youden > 0 else np.nan
        num = tp * tn - fp * fn
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = num / den if den > 0 else np.nan
        row = {"Classe": name, "N_test": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
               "Se": se, "Sp": sp, "VPP": ppv, "VPN": npv, "F1": f1,
               "FPR": fpr, "FNR": fnr, "LR+": lr_pos, "LR-": lr_neg, "DOR": dor,
               "Youden_J": youden, "MCC": mcc, "NND": nnd,
               "CUI+": (ppv * se) if not (np.isnan(ppv) or np.isnan(se)) else np.nan,
               "CUI-": (npv * sp) if not (np.isnan(npv) or np.isnan(sp)) else np.nan}
        for p in prevalence_grid:                   # theoreme de Bayes
            row[f"VPP@prev{int(p*100)}%"] = _div(se * p, se * p + (1 - sp) * (1 - p))
            row[f"VPN@prev{int(p*100)}%"] = _div(sp * (1 - p), sp * (1 - p) + (1 - se) * p)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Statistiques AAMI EC57 : "gross" (tous les battements) et "average" (par patient)
# --------------------------------------------------------------------------- #
def ec57_statistics(y_true, y_pred, groups, class_names, targets=("S", "V")):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    names = list(class_names)
    rows = []
    for target in targets:
        i = names.index(target)
        cm = confusion_matrix(y_true, y_pred, len(names))
        tp, fp, fn, tn = binary_counts(cm, i)
        gross_se, gross_pp = _div(tp, tp + fn), _div(tp, tp + fp)
        se_list, pp_list = [], []
        for g in np.unique(groups):
            m = groups == g
            cmg = confusion_matrix(y_true[m], y_pred[m], len(names))
            t, f_, n_, _ = binary_counts(cmg, i)
            if t + n_ > 0:
                se_list.append(_div(t, t + n_))
            if t + f_ > 0:
                pp_list.append(_div(t, t + f_))
        rows.append({"Classe": target,
                     "Se_gross": gross_se, "PP_gross": gross_pp,
                     "Se_average": float(np.nanmean(se_list)) if se_list else np.nan,
                     "PP_average": float(np.nanmean(pp_list)) if pp_list else np.nan,
                     "Se_min_patient": float(np.nanmin(se_list)) if se_list else np.nan,
                     "Patients_evalues": int(len(se_list))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Charge d'alarme : ce que vit reellement une unite de soins
# --------------------------------------------------------------------------- #
def alarm_burden(y_true, y_pred, groups, durations_s, class_names, target="V"):
    """Fausses alarmes par heure et battements pathologiques manques par heure."""
    names = list(class_names); i = names.index(target)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        hours = durations_s.get(str(g), np.nan) / 3600.0
        cm = confusion_matrix(y_true[m], y_pred[m], len(names))
        tp, fp, fn, _ = binary_counts(cm, i)
        rows.append({"Enregistrement": str(g), "Heures": hours,
                     "Vrais_positifs": tp, "Fausses_alarmes": fp, "Manques": fn,
                     "FA_par_heure": _div(fp, hours), "Manques_par_heure": _div(fn, hours)})
    df = pd.DataFrame(rows)
    total = {"Enregistrement": "GLOBAL", "Heures": df.Heures.sum(),
             "Vrais_positifs": df.Vrais_positifs.sum(),
             "Fausses_alarmes": df.Fausses_alarmes.sum(), "Manques": df.Manques.sum()}
    total["FA_par_heure"] = _div(total["Fausses_alarmes"], total["Heures"])
    total["Manques_par_heure"] = _div(total["Manques"], total["Heures"])
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _trigger_events(flags, times_s, k_of_n=(3, 10), refractory_s=60.0):
    """Regle de declenchement d'un moniteur : k evenements sur une fenetre de n battements,
    suivie d'une periode refractaire pendant laquelle aucune nouvelle alarme n'est levee.

    flags     : sequence booleenne (battement etiquete/predit pathologique)
    times_s   : instants des battements, en secondes
    k_of_n    : (k, n) — au moins k marques parmi n battements consecutifs
    Retourne les instants de declenchement.
    """
    flags = np.asarray(flags, bool); times_s = np.asarray(times_s, float)
    k, n = k_of_n
    events, last = [], -np.inf
    csum = np.concatenate([[0], np.cumsum(flags)])
    for i in range(len(flags) - n + 1):
        if csum[i + n] - csum[i] >= k:
            t = times_s[i + n - 1]
            if t - last >= refractory_s:
                events.append(float(t)); last = t
    return np.asarray(events)


def alarm_burden_debounced(y_true, y_pred, groups, times_s, durations_s, class_names,
                           target="V", k_of_n=(3, 10), refractory_s=60.0, tolerance_s=10.0):
    """Charge d'alarmes AVEC regle de declenchement, comme un moniteur reel.

    Compter une fausse alarme par battement mal classe surestime enormement la charge
    reelle : un Holter ou un moniteur de soins intensifs declenche sur une salve
    (k battements sur n) puis se tait pendant une periode refractaire. Une alarme est
    consideree vraie si un declenchement de reference existe a +/- `tolerance_s`.
    """
    names = list(class_names); i = names.index(target)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    groups = np.asarray(groups); times_s = np.asarray(times_s, float)
    rows = []
    for g in np.unique(groups):
        m = groups == g
        order = np.argsort(times_s[m])
        tt, pp, ts = y_true[m][order], y_pred[m][order], times_s[m][order]
        hours = durations_s.get(str(g), np.nan) / 3600.0
        ref = _trigger_events(tt == i, ts, k_of_n, refractory_s)
        alr = _trigger_events(pp == i, ts, k_of_n, refractory_s)
        if len(alr) and len(ref):
            d = np.abs(alr[:, None] - ref[None, :]).min(axis=1)
            vp = int((d <= tolerance_s).sum())
        else:
            vp = 0
        fa = int(len(alr) - vp)
        if len(ref) and len(alr):
            d2 = np.abs(ref[:, None] - alr[None, :]).min(axis=1)
            manques = int((d2 > tolerance_s).sum())
        else:
            manques = int(len(ref))
        rows.append({"Enregistrement": str(g), "Heures": hours,
                     "Alarmes_reference": int(len(ref)), "Alarmes_emises": int(len(alr)),
                     "Vraies_alarmes": vp, "Fausses_alarmes": fa, "Manquees": manques,
                     "FA_par_heure": _div(fa, hours),
                     "Sensibilite_alarme": _div(vp, len(ref)) if len(ref) else np.nan,
                     "VPP_alarme": _div(vp, len(alr)) if len(alr) else np.nan})
    df = pd.DataFrame(rows)
    tot = {"Enregistrement": "GLOBAL", "Heures": df.Heures.sum(),
           "Alarmes_reference": df.Alarmes_reference.sum(),
           "Alarmes_emises": df.Alarmes_emises.sum(),
           "Vraies_alarmes": df.Vraies_alarmes.sum(),
           "Fausses_alarmes": df.Fausses_alarmes.sum(),
           "Manquees": df.Manquees.sum()}
    tot["FA_par_heure"] = _div(tot["Fausses_alarmes"], tot["Heures"])
    tot["Sensibilite_alarme"] = _div(tot["Vraies_alarmes"], tot["Alarmes_reference"])
    tot["VPP_alarme"] = _div(tot["Vraies_alarmes"], tot["Alarmes_emises"])
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)


# --------------------------------------------------------------------------- #
# Analyse par episode : le clinicien raisonne en salves, pas en battements isoles
# --------------------------------------------------------------------------- #
def _episodes(binary_sequence, min_len=2, max_gap=1):
    """Regroupe les battements positifs en episodes (salves)."""
    idx = np.flatnonzero(binary_sequence)
    if len(idx) == 0:
        return []
    episodes, start, prev = [], idx[0], idx[0]
    for k in idx[1:]:
        if k - prev <= max_gap + 1:
            prev = k
        else:
            if prev - start + 1 >= min_len:
                episodes.append((start, prev))
            start = prev = k
    if prev - start + 1 >= min_len:
        episodes.append((start, prev))
    return episodes


def episode_analysis(y_true, y_pred, groups, class_names, target="V", min_len=2):
    """Sensibilite par episode et latence de detection (en battements)."""
    names = list(class_names); i = names.index(target)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    detected = total = 0
    latencies, false_episodes = [], 0
    for g in np.unique(groups):
        m = np.flatnonzero(groups == g)
        t = (y_true[m] == i).astype(int); p = (y_pred[m] == i).astype(int)
        for a, b in _episodes(t, min_len=min_len):
            total += 1
            hit = np.flatnonzero(p[a:b + 1])
            if len(hit):
                detected += 1
                latencies.append(int(hit[0]))
        for a, b in _episodes(p, min_len=min_len):
            if t[a:b + 1].sum() == 0:
                false_episodes += 1
    return {"Classe": target, "Episodes_reels": total, "Episodes_detectes": detected,
            "Sensibilite_episode": _div(detected, total),
            "Latence_mediane_battements": float(np.median(latencies)) if latencies else np.nan,
            "Episodes_faussement_signales": false_episodes}


# --------------------------------------------------------------------------- #
# Calibration probabiliste
# --------------------------------------------------------------------------- #
def brier_score(y_true, proba, n_classes):
    onehot = np.zeros_like(proba); onehot[np.arange(len(y_true)), np.asarray(y_true, int)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def calibration_errors(y_true, proba, n_bins=15):
    conf = proba.max(1); pred = proba.argmax(1)
    acc = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; mce = 0.0; curve = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            gap = abs(acc[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            mce = max(mce, gap)
            curve.append((float(conf[m].mean()), float(acc[m].mean()), int(m.sum())))
    return {"ECE": float(ece), "MCE": float(mce)}, curve


# --------------------------------------------------------------------------- #
# Analyse de la courbe de decision (net benefit) : rare en ECG, tres apprecie
# --------------------------------------------------------------------------- #
def decision_curve(y_true, proba_target, thresholds=None):
    """Benefice net du modele, de la strategie "traiter tout le monde" et "personne"."""
    y = np.asarray(y_true, int); p = np.asarray(proba_target, float); n = len(y)
    thresholds = np.linspace(0.01, 0.60, 60) if thresholds is None else np.asarray(thresholds)
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        nb_model = tp / n - (fp / n) * (t / (1 - t))
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append({"Seuil": float(t), "NB_modele": nb_model, "NB_traiter_tous": nb_all,
                     "NB_traiter_personne": 0.0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cout clinique
# --------------------------------------------------------------------------- #
def clinical_cost(y_true, y_pred, cost_matrix):
    cm = confusion_matrix(y_true, y_pred, cost_matrix.shape[0])
    total = float((cm * cost_matrix).sum())
    return {"Cout_clinique_total": total, "Cout_moyen_par_battement": total / max(cm.sum(), 1)}


# --------------------------------------------------------------------------- #
# Option de rejet : renvoyer les cas incertains au cardiologue
# --------------------------------------------------------------------------- #
def coverage_risk_curve(y_true, proba, uncertainty=None, steps=21):
    """Compromis couverture / erreur : securite clinique de l'IA."""
    y = np.asarray(y_true, int); pred = proba.argmax(1)
    score = -proba.max(1) if uncertainty is None else np.asarray(uncertainty)
    order = np.argsort(score)                       # du plus sûr au plus incertain
    correct = (pred == y).astype(float)[order]
    rows = []
    for c in np.linspace(0.1, 1.0, steps):
        k = max(1, int(c * len(y)))
        rows.append({"Couverture": float(k / len(y)),
                     "Erreur_selective": float(1 - correct[:k].mean()),
                     "Taux_de_renvoi": float(1 - k / len(y))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Points de fonctionnement cliniques (seuils choisis sur la VALIDATION)
# --------------------------------------------------------------------------- #
def threshold_for_target_sensitivity(y_true_binary, proba_target, target_se):
    """Plus petit seuil garantissant la sensibilite demandee (choisi sur validation)."""
    y = np.asarray(y_true_binary, int); p = np.asarray(proba_target, float)
    pos = p[y == 1]
    if len(pos) == 0:
        return 0.5
    return float(np.quantile(pos, max(0.0, 1.0 - target_se)))


def apply_priority_thresholds(proba, thresholds, priority):
    """Regle de decision clinique : une classe grave l'emporte si son seuil est franchi."""
    pred = proba.argmax(1).copy()
    for cls in priority:                            # ex. [V, S] : V examine en premier
        hit = proba[:, cls] >= thresholds[cls]
        pred[hit] = cls
    return pred


# --------------------------------------------------------------------------- #
# Incertitude bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_metric(y_true, y_pred, fn, n=1000, seed=42, groups=None):
    """IC95 par bootstrap. Si groups est fourni, le tirage se fait PAR PATIENT."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    values = []
    if groups is None:
        idx = np.arange(len(y_true))
        for _ in range(n):
            s = rng.choice(idx, len(idx), replace=True)
            values.append(fn(y_true[s], y_pred[s]))
    else:
        groups = np.asarray(groups); uniq = np.unique(groups)
        pools = {g: np.flatnonzero(groups == g) for g in uniq}
        for _ in range(n):
            picked = rng.choice(uniq, len(uniq), replace=True)
            s = np.concatenate([pools[g] for g in picked])
            values.append(fn(y_true[s], y_pred[s]))
    values = np.asarray(values, float)
    return {"mean": float(np.nanmean(values)),
            "ci95_low": float(np.nanpercentile(values, 2.5)),
            "ci95_high": float(np.nanpercentile(values, 97.5))}


def macro_f1(y_true, y_pred, n_classes=5):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    f1s = []
    for i in range(n_classes):
        tp, fp, fn, _ = binary_counts(cm, i)
        if tp + fn == 0:
            continue
        f1s.append(_div(2 * tp, 2 * tp + fp + fn))
    return float(np.nanmean(f1s)) if f1s else np.nan


def balanced_accuracy(y_true, y_pred, n_classes=5):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    recalls = [_div(cm[i, i], cm[i].sum()) for i in range(n_classes) if cm[i].sum() > 0]
    return float(np.nanmean(recalls)) if recalls else np.nan


def mcnemar_test(y_true, pred_a, pred_b):
    """Test de McNemar avec correction de continuite (comparaison appariee)."""
    a = (np.asarray(pred_a) == np.asarray(y_true))
    b = (np.asarray(pred_b) == np.asarray(y_true))
    n01 = int(np.sum(a & ~b)); n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n01": n01, "n10": n10}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    # p-value chi2 a 1 ddl = erfc(sqrt(chi2/2))
    from math import erfc, sqrt
    return {"chi2": float(chi2), "p_value": float(erfc(sqrt(chi2 / 2))), "n01": n01, "n10": n10}
