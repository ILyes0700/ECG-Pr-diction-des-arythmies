"""Agregation des resultats : toutes les tables de l'article (numpy pur)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from .clinical_metrics import (confusion_matrix, binary_counts, clinical_table, ec57_statistics,
                               alarm_burden, episode_analysis, brier_score, calibration_errors,
                               decision_curve, clinical_cost, coverage_risk_curve,
                               bootstrap_metric, macro_f1, balanced_accuracy, mcnemar_test,
                               threshold_for_target_sensitivity, apply_priority_thresholds)


# --------------------------------------------------------------------------- #
# Courbes (implementations numpy, sans scikit-learn)
# --------------------------------------------------------------------------- #
def roc_curve_np(y_binary, score):
    y = np.asarray(y_binary, int); s = np.asarray(score, float)
    order = np.argsort(-s); y = y[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    P, N = max(y.sum(), 1), max(len(y) - y.sum(), 1)
    return np.concatenate([[0], fps / N, [1]]), np.concatenate([[0], tps / P, [1]])


def pr_curve_np(y_binary, score):
    y = np.asarray(y_binary, int); s = np.asarray(score, float)
    order = np.argsort(-s); y = y[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max(y.sum(), 1)
    return recall, precision


def auc_np(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order])) if hasattr(np, "trapezoid") \
        else float(np.trapz(y[order], x[order]))


def macro_auc(y_true, proba):
    aucs = []
    for i in range(proba.shape[1]):
        yb = (np.asarray(y_true) == i).astype(int)
        if yb.sum() == 0:
            continue
        fpr, tpr = roc_curve_np(yb, proba[:, i])
        aucs.append(auc_np(fpr, tpr))
    return float(np.mean(aucs)) if aucs else np.nan


def macro_ap(y_true, proba):
    aps = []
    for i in range(proba.shape[1]):
        yb = (np.asarray(y_true) == i).astype(int)
        if yb.sum() == 0:
            continue
        rec, prec = pr_curve_np(yb, proba[:, i])
        aps.append(auc_np(rec, prec))
    return float(np.mean(aps)) if aps else np.nan


def macro_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, C.N_CLASSES)
    vals = []
    for i in range(C.N_CLASSES):
        tp, fp, fn, tn = binary_counts(cm, i)
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if den > 0:
            vals.append((tp * tn - fp * fn) / den)
    return float(np.mean(vals)) if vals else np.nan


def cohen_kappa(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, C.N_CLASSES).astype(float)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


# --------------------------------------------------------------------------- #
# Metriques globales d'un run
# --------------------------------------------------------------------------- #
def global_metrics(y_true, proba, y_pred=None):
    y_true = np.asarray(y_true)
    pred = proba.argmax(1) if y_pred is None else np.asarray(y_pred)
    ct = clinical_table(y_true, pred, C.CLASS_NAMES)
    errs, _ = calibration_errors(y_true, proba)
    out = {
        "Accuracy": float((pred == y_true).mean()),
        "Balanced_Acc": balanced_accuracy(y_true, pred, C.N_CLASSES),
        "F1_macro": macro_f1(y_true, pred, C.N_CLASSES),
        "MCC_macro": macro_mcc(y_true, pred),
        "Kappa": cohen_kappa(y_true, pred),
        "AUC_macro": macro_auc(y_true, proba),
        "AP_macro": macro_ap(y_true, proba),
        "Brier": brier_score(y_true, proba, C.N_CLASSES),
        "ECE": errs["ECE"], "MCE": errs["MCE"],
    }
    for cls in ["S", "V", "F"]:
        r = ct[ct.Classe == cls].iloc[0]
        out[f"Se_{cls}"] = r["Se"]; out[f"PP_{cls}"] = r["VPP"]; out[f"F1_{cls}"] = r["F1"]
    out.update(clinical_cost(y_true, pred, C.COST_MATRIX))
    return out, ct


# --------------------------------------------------------------------------- #
# Tables de l'article
# --------------------------------------------------------------------------- #
def table_dataset(y, g):
    rows = []
    for label, recs in [("DS1 (train + validation)", C.DS1), ("DS2 (test)", C.DS2)]:
        m = np.isin(g, recs)
        counts = np.bincount(y[m], minlength=C.N_CLASSES)
        rows.append({"Ensemble": label, "Patients": int(len(np.unique(g[m]))),
                     **{c: int(v) for c, v in zip(C.CLASS_NAMES, counts)},
                     "Total": int(m.sum())})
    counts = np.bincount(y, minlength=C.N_CLASSES)
    rows.append({"Ensemble": "Total", "Patients": int(len(np.unique(g))),
                 **{c: int(v) for c, v in zip(C.CLASS_NAMES, counts)}, "Total": int(len(y))})
    return pd.DataFrame(rows)


def table_comparison(runs, models):
    metrics = ["Accuracy", "Balanced_Acc", "F1_macro", "MCC_macro", "Kappa", "AUC_macro",
               "Se_S", "PP_S", "Se_V", "PP_V", "Brier", "ECE", "Params", "Latence_ms"]
    rows = []
    for m in models:
        d = runs[runs.Model == m]
        if d.empty:
            continue
        row = {"Model": m, "Runs": len(d)}
        for k in metrics:
            if k not in d:
                continue
            mu, sd = d[k].mean(), d[k].std(ddof=0)
            row[k] = f"{mu:,.0f}" if k == "Params" else (
                f"{mu:.4f} ± {sd:.4f}" if len(d) > 1 else f"{mu:.4f}")
            row[k + "_val"] = float(mu)
        rows.append(row)
    return pd.DataFrame(rows)


def table_operating_point(y_val, proba_val, y_test, proba_test):
    """Seuils cliniques choisis sur la VALIDATION, appliques une seule fois au test."""
    thresholds = {i: 1.1 for i in range(C.N_CLASSES)}
    thresholds[C.IDX_V] = threshold_for_target_sensitivity(
        (np.asarray(y_val) == C.IDX_V).astype(int), proba_val[:, C.IDX_V], C.TARGET_SE_V)
    thresholds[C.IDX_S] = threshold_for_target_sensitivity(
        (np.asarray(y_val) == C.IDX_S).astype(int), proba_val[:, C.IDX_S], C.TARGET_SE_S)
    pred_argmax = proba_test.argmax(1)
    pred_clin = apply_priority_thresholds(proba_test, thresholds, priority=[C.IDX_S, C.IDX_V])
    rows = []
    for label, pred in [("argmax (standard)", pred_argmax),
                        ("point de fonctionnement clinique", pred_clin)]:
        ct = clinical_table(y_test, pred, C.CLASS_NAMES)
        rows.append({
            "Regle de decision": label,
            "Se_V": ct[ct.Classe == "V"].Se.iloc[0], "VPP_V": ct[ct.Classe == "V"].VPP.iloc[0],
            "Se_S": ct[ct.Classe == "S"].Se.iloc[0], "VPP_S": ct[ct.Classe == "S"].VPP.iloc[0],
            "F1_macro": macro_f1(y_test, pred, C.N_CLASSES),
            "Accuracy": float((pred == np.asarray(y_test)).mean()),
        })
    return pd.DataFrame(rows), thresholds, pred_clin


def table_per_patient(y_true, y_pred, groups):
    rows = []
    for g in sorted(set(np.asarray(groups).tolist())):
        m = np.asarray(groups) == g
        rows.append({"Enregistrement": g, "Battements": int(m.sum()),
                     "Accuracy": float((np.asarray(y_pred)[m] == np.asarray(y_true)[m]).mean()),
                     "F1_macro": macro_f1(np.asarray(y_true)[m], np.asarray(y_pred)[m], C.N_CLASSES)})
    return pd.DataFrame(rows)


def table_significance(y_true, proba_best, probas_baselines, n_boot=C.BOOTSTRAP_N, groups=None):
    pred_best = proba_best.argmax(1)
    ci_best = bootstrap_metric(y_true, pred_best, lambda a, b: macro_f1(a, b, C.N_CLASSES),
                               n=n_boot, groups=groups)
    rows = []
    for name, proba in probas_baselines.items():
        pred = proba.argmax(1)
        ci = bootstrap_metric(y_true, pred, lambda a, b: macro_f1(a, b, C.N_CLASSES),
                              n=n_boot, groups=groups)
        mn = mcnemar_test(y_true, pred_best, pred)
        rows.append({"Baseline": name,
                     "F1_macro_baseline": f"{ci['mean']:.4f} [{ci['ci95_low']:.4f}–{ci['ci95_high']:.4f}]",
                     "F1_macro_propose": f"{ci_best['mean']:.4f} [{ci_best['ci95_low']:.4f}–{ci_best['ci95_high']:.4f}]",
                     "Delta": round(ci_best["mean"] - ci["mean"], 4),
                     "McNemar_chi2": round(mn["chi2"], 2),
                     "p": f"{mn['p_value']:.2e}",
                     "Significatif_5%": "oui" if mn["p_value"] < 0.05 else "non"})
    return pd.DataFrame(rows), ci_best


def table_literature():
    """Comparaison a l'etat de l'art inter-patient (valeurs publiees)."""
    return pd.DataFrame([
        {"Travaux": "de Chazal et al., IEEE TBME 2004", "Protocole": "inter-patient DS1/DS2",
         "Se_S": 0.757, "VPP_S": 0.387, "Se_V": 0.777, "VPP_V": 0.816, "Accuracy": 0.856},
        {"Travaux": "Mar et al., IEEE TITB 2011", "Protocole": "inter-patient DS1/DS2",
         "Se_S": 0.833, "VPP_S": 0.331, "Se_V": 0.866, "VPP_V": 0.751, "Accuracy": 0.899},
        {"Travaux": "Zhang et al., 2014 (SVM + RR)", "Protocole": "inter-patient DS1/DS2",
         "Se_S": 0.791, "VPP_S": 0.361, "Se_V": 0.854, "VPP_V": 0.929, "Accuracy": 0.867},
        {"Travaux": "Sellami & Hwang, ESWA 2019 (CNN)", "Protocole": "inter-patient DS1/DS2",
         "Se_S": 0.821, "VPP_S": 0.622, "Se_V": 0.947, "VPP_V": 0.921, "Accuracy": 0.882},
        {"Travaux": "Wang et al., 2021 (CNN + RR)", "Protocole": "inter-patient DS1/DS2",
         "Se_S": 0.848, "VPP_S": 0.741, "Se_V": 0.941, "VPP_V": 0.934, "Accuracy": 0.933},
    ])


def add_result_row(y_true, proba, model_name, seed, params, latency_ms, train_time,
                   variant="inter-patient"):
    g, _ = global_metrics(y_true, proba)
    g.update({"Model": model_name, "Seed": seed, "Params": params,
              "Latence_ms": latency_ms, "Train_s": train_time, "Variante": variant})
    return g
