"""Toutes les figures de l'article (14 figures, 330 dpi, style publication)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as C
from .clinical_metrics import confusion_matrix, calibration_errors

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 330, "savefig.bbox": "tight",
                     "axes.titleweight": "bold", "font.family": "DejaVu Sans"})
PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print("  figure ->", path.name)
    return path


# 1 ------------------------------------------------------------------ signaux
def fig_signal_examples(X, y, name="fig01_exemples_battements.png"):
    fig, axes = plt.subplots(1, C.N_CLASSES, figsize=(16, 2.8), sharey=True)
    for i, cls in enumerate(C.CLASS_NAMES):
        idx = np.flatnonzero(y == i)
        ax = axes[i]
        for k in idx[:25]:
            ax.plot(X[k, :, 1], color=PALETTE[i], alpha=0.25, lw=0.8)
        if len(idx):
            ax.plot(X[idx[:200], :, 1].mean(0), color="black", lw=2)
        ax.set_title(f"Classe {cls}  (n={len(idx):,})")
        ax.set_xlabel("Echantillon")
    axes[0].set_ylabel("Amplitude (z-score)")
    fig.suptitle("Morphologies moyennes par classe AAMI", y=1.06)
    return _save(fig, name)


# 2 ------------------------------------------------------------- distribution
def fig_class_distribution(table1, name="fig02_distribution.png"):
    d = table1[table1.Ensemble != "Total"].melt(id_vars="Ensemble",
                                                value_vars=list(C.CLASS_NAMES),
                                                var_name="Classe", value_name="Battements")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.barplot(data=d, x="Classe", y="Battements", hue="Ensemble", ax=ax, palette="Set2")
    ax.set_yscale("log")
    ax.set_title("Distribution des battements (echelle log) — desequilibre 1:100")
    return _save(fig, name)


# 3 ------------------------------------------------- courbes d'apprentissage
def fig_learning_curves(histories, name="fig03_apprentissage.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (model_name, seed), h in histories.items():
        if seed != list(histories.keys())[0][1]:
            continue
        axes[0].plot(h["loss"], label=f"{model_name} (train)", lw=1.6)
        axes[0].plot(h["val_loss"], "--", label=f"{model_name} (val)", lw=1.6)
        axes[1].plot(h["accuracy"], lw=1.6)
        axes[1].plot(h["val_accuracy"], "--", lw=1.6)
    axes[0].set_title("Perte"); axes[1].set_title("Exactitude")
    for ax in axes:
        ax.set_xlabel("Epoque")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Courbes d'apprentissage (train continu / validation pointille)", y=1.03)
    return _save(fig, name)


# 4 -------------------------------------------------- matrices de confusion
def fig_confusion(y_true, y_pred, title, name):
    cm = confusion_matrix(y_true, y_pred, C.N_CLASSES)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f"{title} — effectifs")
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1,
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES, ax=axes[1])
    axes[1].set_title(f"{title} — sensibilite par ligne")
    for ax in axes:
        ax.set_xlabel("Predit"); ax.set_ylabel("Reel")
    return _save(fig, name)


# 5 ------------------------------------------------------------------ ROC/PR
def fig_roc_pr(y_true, proba, name="fig05_roc_pr.png"):
    from .evaluate import roc_curve_np, pr_curve_np, auc_np
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, cls in enumerate(C.CLASS_NAMES):
        yb = (np.asarray(y_true) == i).astype(int)
        fpr, tpr = roc_curve_np(yb, proba[:, i])
        rec, prec = pr_curve_np(yb, proba[:, i])
        axes[0].plot(fpr, tpr, lw=2, color=PALETTE[i], label=f"{cls} (AUC={auc_np(fpr, tpr):.3f})")
        axes[1].plot(rec, prec, lw=2, color=PALETTE[i], label=f"{cls} (AP={auc_np(rec, prec):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    axes[0].set_xlabel("1 - Specificite"); axes[0].set_ylabel("Sensibilite")
    axes[0].set_title("Courbes ROC (un contre tous)")
    axes[1].set_xlabel("Rappel (sensibilite)"); axes[1].set_ylabel("Precision (VPP)")
    axes[1].set_title("Courbes precision-rappel")
    for ax in axes:
        ax.legend(fontsize=8)
    return _save(fig, name)


# 6 ---------------------------------------------------------- comparaison
def fig_model_comparison(runs, name="fig06_comparaison.png"):
    metrics = ["F1_macro", "Balanced_Acc", "Se_S", "Se_V", "MCC_macro"]
    d = runs.melt(id_vars="Model", value_vars=[m for m in metrics if m in runs],
                  var_name="Metrique", value_name="Score")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    sns.barplot(data=d, x="Metrique", y="Score", hue="Model", errorbar="sd", ax=ax)
    ax.set_ylim(0, 1.02)
    ax.set_title("Comparaison inter-patient sur DS2 (moyenne ± ecart-type des seeds)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    return _save(fig, name)


# 7 ------------------------------------------------------------ calibration
def fig_calibration(y_true, proba_raw, proba_cal, temperature, name="fig07_calibration.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, proba, title in [(axes[0], proba_raw, "Avant calibration"),
                             (axes[1], proba_cal, f"Apres (T={temperature:.2f})")]:
        errs, curve = calibration_errors(y_true, proba)
        xs = [c[0] for c in curve]; ys = [c[1] for c in curve]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Calibration parfaite")
        ax.plot(xs, ys, "o-", lw=2, color="#e76f51")
        ax.set_title(f"{title}\nECE={errs['ECE']:.4f} | MCE={errs['MCE']:.4f}")
        ax.set_xlabel("Confiance moyenne"); ax.set_ylabel("Exactitude observee")
        ax.legend(fontsize=8)
    axes[2].hist(proba_raw.max(1), bins=40, alpha=.6, label="avant", color="#264653")
    axes[2].hist(proba_cal.max(1), bins=40, alpha=.6, label="apres", color="#2a9d8f")
    axes[2].set_title("Distribution de la confiance"); axes[2].legend()
    return _save(fig, name)


# 8 ------------------------------------------------- courbe de decision
def fig_decision_curve(dc, name="fig08_decision_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(dc.Seuil, dc.NB_modele, lw=2.5, color="#2a9d8f", label="Modele propose")
    ax.plot(dc.Seuil, dc.NB_traiter_tous, "--", lw=1.8, color="#e76f51", label="Alerter tous")
    ax.plot(dc.Seuil, dc.NB_traiter_personne, ":", lw=1.8, color="gray", label="N'alerter personne")
    ax.set_xlabel("Seuil de probabilite (preference clinique)")
    ax.set_ylabel("Benefice net")
    ax.set_title("Analyse de la courbe de decision — detection des battements V")
    ax.legend()
    return _save(fig, name)


# 9 ------------------------------------------------- couverture / risque
def fig_coverage_risk(crc, deferral, name="fig09_couverture_risque.png"):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(crc.Couverture, crc.Erreur_selective, "o-", lw=2, color="#264653")
    ax.axvline(deferral["couverture"], ls="--", color="#e76f51",
               label=f"99 % de fiabilite a {deferral['couverture']*100:.0f} % de couverture")
    ax.set_xlabel("Couverture (part des battements decides par l'IA)")
    ax.set_ylabel("Taux d'erreur selectif")
    ax.set_title("Option de rejet : securite clinique de la decision automatique")
    ax.legend(fontsize=8)
    return _save(fig, name)


# 10 --------------------------------------------------------- alarmes/heure
def fig_alarm_burden(alarms, name="fig10_alarmes.png"):
    d = alarms[alarms.Enregistrement != "GLOBAL"].sort_values("FA_par_heure")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(d.Enregistrement, d.FA_par_heure, color="#e76f51", label="Fausses alarmes / h")
    ax.bar(d.Enregistrement, -d.Manques_par_heure, color="#264653", label="Battements V manques / h")
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("Evenements par heure"); ax.set_xlabel("Enregistrement DS2")
    ax.set_title("Charge d'alarme par patient (fatigue d'alarme en unite de soins)")
    ax.legend(); plt.setp(ax.get_xticklabels(), rotation=45)
    return _save(fig, name)


# 11 ----------------------------------------------------------- par patient
def fig_per_patient(per_patient, name="fig11_par_patient.png"):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    d = per_patient.sort_values("F1_macro")
    sns.barplot(data=d, x="Enregistrement", y="F1_macro", color="#2a9d8f", ax=ax)
    ax.axhline(d.F1_macro.mean(), ls="--", color="red",
               label=f"moyenne = {d.F1_macro.mean():.3f}")
    ax.set_title("Variabilite inter-patient sur DS2"); ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45)
    return _save(fig, name)


# 12 -------------------------------------------------------------- ablation
def fig_ablation(table_ablation, name="fig12_ablation.png"):
    d = table_ablation.copy()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    sns.barplot(data=d, y="Configuration", x="F1_macro_val", color="#264653", ax=ax)
    ax.set_xlabel("F1 macro (DS2)"); ax.set_ylabel("")
    ax.set_title("Etude d'ablation : contribution de chaque composant")
    return _save(fig, name)


# 13 ------------------------------------------------------- interpretabilite
def fig_interpretability(X, y, y_pred, attributions, name="fig13_interpretabilite.png"):
    fig, axes = plt.subplots(len(attributions), 1, figsize=(11, 2.4 * len(attributions)))
    axes = np.atleast_1d(axes)
    for ax, (idx, attr, cls) in zip(axes, attributions):
        ax.plot(X[idx, :, 1], color="#264653", lw=1.7, label="ECG (z-score)")
        ax2 = ax.twinx()
        ax2.fill_between(np.arange(X.shape[1]), np.abs(attr), color="#e76f51", alpha=.35)
        ax2.set_yticks([])
        ax.set_title(f"Classe reelle {C.CLASS_NAMES[cls]} — prediction "
                     f"{C.CLASS_NAMES[y_pred[idx]]} — attribution integree")
        ax.set_xlabel("Echantillon")
    fig.tight_layout()
    return _save(fig, name)


# 14 ---------------------------------------------------------- complexite
def fig_complexity(comp, name="fig14_complexite.png"):
    fig, ax = plt.subplots(figsize=(6.8, 5))
    ax.scatter(comp.Params / 1e3, comp.F1_macro, s=160,
               c=np.arange(len(comp)), cmap="viridis", zorder=3)
    for _, r in comp.iterrows():
        ax.annotate(r.Model, (r.Params / 1e3, r.F1_macro),
                    textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlabel("Parametres (milliers)"); ax.set_ylabel("F1 macro (DS2)")
    ax.set_title("Compromis performance / complexite")
    return _save(fig, name)


# 16 ------------------------------------------------------- controle qualite
def fig_quality(X, keep, q, name="fig16_qualite.png"):
    """Distribution des indices de qualite et exemples de battements rejetes."""
    import numpy as _np
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(q["kSQI"], bins=60, color="#264653")
    ax.axvline(2.0, color="red", ls="--", label="seuil kSQI = 2")
    ax.set_title("Indice de qualite kSQI (kurtosis)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(q["hf_ratio"], bins=60, color="#2a9d8f")
    ax.axvline(0.6, color="red", ls="--", label="seuil bruit HF = 0,6")
    ax.set_title("Energie haute frequence (bruit EMG)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(["conserves", "rejetes"], [int(keep.sum()), int((~keep).sum())],
           color=["#2a9d8f", "#e76f51"])
    ax.set_title(f"Battements ({100*(~keep).mean():.2f} % rejetes)")

    rejected = _np.flatnonzero(~keep)[:3]
    kept = _np.flatnonzero(keep)[:3]
    for j, (idx_list, titre, couleur) in enumerate(
            [(kept, "conserve", "#2a9d8f"), (rejected, "rejete", "#e76f51")]):
        ax = fig.add_subplot(gs[1, j])
        for k in idx_list:
            sig = X[k, :, 1] if X.ndim == 3 else X[k]
            ax.plot(sig, color=couleur, alpha=.8, lw=1.3)
        ax.set_title(f"Exemples : {titre}")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    ax.text(0.0, 0.5, "Nettoyage applique au TRAIN uniquement.\n"
                      "Aucun battement n'est retire du TEST :\n"
                      "il est seulement compte et rapporte.",
            fontsize=11, va="center")
    fig.suptitle("Controle qualite du signal", y=1.02)
    fig.tight_layout()
    return _save(fig, name)
