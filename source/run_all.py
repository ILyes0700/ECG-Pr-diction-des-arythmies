"""Reproduit l'integralite des resultats de l'article.

    python run_all.py --data data/mitdb --quick          # mise au point (~30 min GPU)
    python run_all.py --data data/mitdb                  # run final 5 seeds (~5 h GPU)
    python run_all.py --data data/mitdb --external data/incartdb

Tout est ecrit dans results/ : tables/ (CSV + LaTeX), figures/ (PNG 330 dpi),
models/ et summary.json.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from src import config as C


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/mitdb", help="dossier MIT-BIH (.dat/.hea/.atr)")
    p.add_argument("--external", default=None, help="dossier de validation externe (INCART/SVDB)")
    p.add_argument("--quick", action="store_true", help="mode mise au point")
    p.add_argument("--no-adaptation", action="store_true", help="desactive l'adaptation patient")
    p.add_argument("--models", default=None, help="liste separee par des virgules")
    p.add_argument("--no-resume", action="store_true",
                   help="recalcule tout meme si des resultats sont deja en cache")
    return p.parse_args()


def save_table(df, stem, caption, label, float_format="%.4f"):
    df.to_csv(C.TAB_DIR / f"{stem}.csv", index=False)
    try:
        (C.TAB_DIR / f"{stem}.tex").write_text(
            df.to_latex(index=False, escape=False, float_format=float_format,
                        caption=caption, label=label), encoding="utf-8")
    except Exception as exc:
        print("  [latex]", exc)
    print(f"  table -> {stem}.csv / .tex")


def main():
    args = parse_args()
    if args.quick:
        C.apply_quick_run()
    C.ensure_dirs()

    from src.datasets import extract_records, build_split, test_durations_after_adaptation
    from src.models import BUILDERS
    from src.train import train_model, adapt_to_patients, predict_proba, get_logits_model, set_all_seeds
    from src import evaluate as E
    from src import plots as P
    from src.calibration import (apply_temperature, proba_to_logits, fit_temperature,
                                 softmax, calibrate_ensemble)
    from src.uncertainty import predictive_entropy, deferral_for_target_accuracy
    from src.clinical_metrics import (clinical_table, ec57_statistics, alarm_burden,
                                      episode_analysis, decision_curve, coverage_risk_curve,
                                      calibration_errors, brier_score)

    t_start = time.time()
    use_adapt = not args.no_adaptation

    # --------------------------------------------------------------- donnees
    print("\n[1/9] Extraction des battements")
    records = sorted(C.DS1 + C.DS2)
    X, RR, y, g, t, durations = extract_records(args.data, records, cache_name="mitdb_ds1ds2")
    print(f"  X={X.shape}  classes={dict(zip(*np.unique(y, return_counts=True)))}")

    # Un cache produit avant le passage a 4 classes peut contenir l'ancienne classe Q.
    valid = y < C.N_CLASSES
    if not valid.all():
        print(f"  classe Q retiree du cache : {int((~valid).sum())} battements ecartes")
        X, RR, y, g, t = X[valid], RR[valid], y[valid], g[valid], t[valid]

    split = build_split(X, RR, y, g, t, durations, use_adaptation=use_adapt)
    eval_durations = test_durations_after_adaptation(durations, use_adapt)

    # ---------------------------------------------------- controle qualite
    from src.quality import quality_report, clean_training_set, quality_mask
    print("\n[1b/9] Controle qualite du signal")
    rep_tr, keep_tr = quality_report(split.Xtr, split.ytr, C.CLASS_NAMES, "train")
    rep_te, _ = quality_report(split.Xte, split.yte, C.CLASS_NAMES, "test (rapporte, non filtre)")
    qtable = pd.concat([rep_tr, rep_te], ignore_index=True)
    save_table(qtable, "table00_qualite_signal",
               "Controle qualite du signal (nettoyage applique au train uniquement).",
               "tab:quality", "%.2f")
    _, q_all = quality_mask(split.Xtr)
    P.fig_quality(split.Xtr, keep_tr, q_all)

    Xtr_c, Rtr_c, ytr_c, gtr_c, _ = clean_training_set(split.Xtr, split.Rtr, split.ytr, split.gtr)
    split.Xtr, split.Rtr, split.ytr, split.gtr = Xtr_c, Rtr_c, ytr_c, gtr_c
    from src.datasets import compute_class_weights
    split.class_weights = compute_class_weights(
        np.bincount(split.ytr, minlength=C.N_CLASSES))
    print("  poids de classes recalcules apres nettoyage :",
          {C.CLASS_NAMES[k]: round(v, 3) for k, v in split.class_weights.items()})

    table1 = E.table_dataset(y, g)
    save_table(table1, "table01_dataset", "Composition de la base MIT-BIH (AAMI, records stimules exclus).", "tab:dataset", "%.0f")
    P.fig_signal_examples(split.Xtr, split.ytr)
    P.fig_class_distribution(table1)

    # ---------------------------------------------------------- entrainement
    print("\n[2/9] Entrainement")
    model_list = args.models.split(",") if args.models else (C.MODELS_MAIN + C.MODELS_ABLATION)
    runs, proba_test, proba_val, histories, keep_models = [], {}, {}, {}, {}

    PROBA_DIR = C.OUT_DIR / "proba"; PROBA_DIR.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    for name in model_list:
        proba_test[name], proba_val[name] = [], []
        for seed in C.SEEDS:
            cache_npz = PROBA_DIR / f"{name}_seed{seed}.npz"
            ckpt = C.MOD_DIR / f"{name}_seed{seed}.keras"

            # --- reprise apres deconnexion Colab : on ne recalcule pas l'existant ----
            if resume and cache_npz.exists():
                z = np.load(cache_npz, allow_pickle=True)
                pte, pva = z["pte"], z["pva"]
                proba_test[name].append(pte); proba_val[name].append(pva)
                runs.append(E.add_result_row(split.yte, pte, name, seed,
                                             int(z["params"]), float(z["latency"]),
                                             float(z["train_time"])))
                print(f"  [reprise] {name} | seed {seed} deja calcule "
                      f"(F1 macro = {runs[-1]['F1_macro']:.4f})")
                if seed == C.SEEDS[0] and ckpt.exists():
                    try:
                        import tensorflow as tf
                        keep_models[name] = tf.keras.models.load_model(ckpt, compile=False)
                    except Exception as exc:
                        print("   modele non rechargeable :", exc)
                continue

            print(f"\n  --- {name} | seed {seed} ---")
            model, hist, ttime = train_model(name, split, seed)
            histories[(name, seed)] = hist

            model_adapted = adapt_to_patients(model, split, seed, model_name=name) \
                if (use_adapt and name == "PROPOSED") else model

            t0 = time.time()
            pte = predict_proba(model_adapted, split.Xte, split.Rte)
            latency = 1000 * (time.time() - t0) / max(len(split.yte), 1)
            pva = predict_proba(model, split.Xva, split.Rva)

            np.savez_compressed(cache_npz, pte=pte, pva=pva, params=model.count_params(),
                                latency=round(latency, 4), train_time=round(ttime, 1),
                                history=json.dumps({k: [float(x) for x in v]
                                                    for k, v in hist.items()}))
            proba_test[name].append(pte); proba_val[name].append(pva)
            runs.append(E.add_result_row(split.yte, pte, name, seed, model.count_params(),
                                         round(latency, 4), round(ttime, 1)))
            print("   ", {k: round(v, 4) for k, v in runs[-1].items()
                          if k in ("F1_macro", "Balanced_Acc", "Se_S", "Se_V", "Accuracy")})
            if seed == C.SEEDS[0]:
                keep_models[name] = model_adapted

    runs = pd.DataFrame(runs)
    runs.to_csv(C.TAB_DIR / "runs_raw.csv", index=False)

    # -------------------------------------------------- ensemble + calibration
    print("\n[3/9] Ensemble des seeds et calibration")
    BEST = "PROPOSED" if "PROPOSED" in proba_test else model_list[0]
    p_raw = np.mean(proba_test[BEST], axis=0)
    p_raw_val = np.mean(proba_val[BEST], axis=0)

    # Calibration de la MOYENNE D'ENSEMBLE (et non d'un seul seed) : temperature scaling
    # contre vector scaling, la methode retenue est celle de plus faible NLL sur la
    # VALIDATION. Le vector scaling corrige le decalage de prior induit par la
    # ponderation des classes, ce que la temperature seule ne peut pas faire.

    p_cal, p_cal_val, calib_info = calibrate_ensemble(p_raw_val, split.yva, p_raw)
    T = calib_info["temperature"]
    print("  calibration retenue :", calib_info["methode"],
          f"| NLL val {calib_info['nll_sans_calibration']:.4f} -> {calib_info['nll_validation']:.4f}")

    # Toutes les metriques en aval utilisent les probabilites CALIBREES.
    p_ens, p_ens_val = p_cal, p_cal_val
    P.fig_calibration(split.yte, p_raw, p_cal, T)

    # ------------------------------------------------------------- tables
    print("\n[4/9] Tables de resultats")
    table2 = E.table_comparison(runs, [m for m in C.MODELS_MAIN if m in set(runs.Model)])
    save_table(table2.drop(columns=[c for c in table2 if c.endswith('_val')]),
               "table02_comparaison", "Comparaison inter-patient sur DS2.", "tab:comparison", "%s")

    table3 = clinical_table(split.yte, p_ens.argmax(1), C.CLASS_NAMES)
    save_table(table3, "table03_metriques_cliniques",
               "Metriques diagnostiques par classe AAMI (Se, Sp, VPP, VPN, LR+, LR-, DOR, MCC, NND).",
               "tab:clinical")

    prev_cols = ["Classe"] + [c for c in table3.columns if "@prev" in c]
    save_table(table3[prev_cols], "table04_prevalence",
               "VPP et VPN ajustes a differentes prevalences (theoreme de Bayes).", "tab:prevalence")

    table5 = ec57_statistics(split.yte, p_ens.argmax(1), split.gte, C.CLASS_NAMES)
    save_table(table5, "table05_ec57", "Statistiques AAMI EC57 gross et average.", "tab:ec57")

    alarms = alarm_burden(split.yte, p_ens.argmax(1), split.gte, eval_durations, C.CLASS_NAMES)
    # Charge d'alarmes realiste : un moniteur ne declenche pas a chaque battement mal
    # classe, mais sur une salve (k parmi n) suivie d'une periode refractaire.
    from src.clinical_metrics import alarm_burden_debounced
    alarms_db = alarm_burden_debounced(split.yte, p_ens.argmax(1), split.gte, split.tte,
                                       eval_durations, C.CLASS_NAMES, target="V",
                                       k_of_n=(3, 10), refractory_s=60.0, tolerance_s=10.0)
    save_table(alarms_db, "table06b_alarmes_declenchement",
               "Charge d'alarmes avec regle de declenchement (3 battements V sur 10, "
               "periode refractaire de 60 s) : comptage par alarme et non par battement.",
               "tab:alarms_debounced")
    save_table(alarms, "table06_alarmes", "Charge d'alarme par enregistrement.", "tab:alarms")
    P.fig_alarm_burden(alarms)

    episodes = pd.DataFrame([episode_analysis(split.yte, p_ens.argmax(1), split.gte, C.CLASS_NAMES, tgt)
                             for tgt in ("V", "S")])
    save_table(episodes, "table07_episodes", "Detection par episode et latence.", "tab:episodes")

    errs, _ = calibration_errors(split.yte, p_cal)
    table8 = pd.DataFrame([
        {"Etat": "sans calibration", "Methode": "-",
         **calibration_errors(split.yte, p_raw)[0],
         "Brier": brier_score(split.yte, p_raw, C.N_CLASSES),
         "NLL_validation": calib_info["nll_sans_calibration"]},
        {"Etat": "temperature scaling", "Methode": f"T={calib_info['temperature']:.2f}",
         **calibration_errors(split.yte, apply_temperature(proba_to_logits(p_raw), calib_info["temperature"]))[0],
         "Brier": brier_score(split.yte, apply_temperature(proba_to_logits(p_raw), calib_info["temperature"]), C.N_CLASSES),
         "NLL_validation": calib_info["nll_temperature"]},
        {"Etat": "retenue (NLL validation)", "Methode": calib_info["methode"], **errs,
         "Brier": brier_score(split.yte, p_cal, C.N_CLASSES),
         "NLL_validation": calib_info["nll_validation"]},
    ])
    save_table(table8, "table08_calibration",
               "Calibration probabiliste : la methode est choisie sur la validation, jamais sur le test.",
               "tab:calibration")

    op_table, thresholds, pred_clin = E.table_operating_point(split.yva, p_ens_val, split.yte, p_ens)
    save_table(op_table, "table09_point_de_fonctionnement",
               "Points de fonctionnement : argmax contre seuils cliniques fixes sur la validation.",
               "tab:operating")

    ent = predictive_entropy(p_cal)
    crc = coverage_risk_curve(split.yte, p_cal, ent)
    deferral = deferral_for_target_accuracy(split.yte, p_cal, ent, 0.99)
    save_table(crc, "table10_couverture_risque", "Compromis couverture / erreur avec option de rejet.", "tab:coverage")
    P.fig_coverage_risk(crc, deferral)

    abl_models = [m for m in (["PROPOSED"] + C.MODELS_ABLATION) if m in set(runs.Model)]
    table11 = E.table_comparison(runs, abl_models)[["Model", "F1_macro", "F1_macro_val",
                                                    "Balanced_Acc", "Se_S", "Se_V"]]
    labels = {"PROPOSED": "Modele complet", "ABL_noRR": "sans descripteurs RR",
              "ABL_noCTX": "sans contexte 3 battements", "ABL_noATT": "sans self-attention",
              "ABL_noSE": "sans blocs SE", "ABL_CE": "perte CE ponderee",
              "ABL_noAUG": "sans augmentation"}
    table11.insert(0, "Configuration", table11.Model.map(labels).fillna(table11.Model))
    save_table(table11.drop(columns=["Model", "F1_macro_val"]), "table11_ablation",
               "Etude d'ablation sur DS2.", "tab:ablation", "%s")
    P.fig_ablation(table11)

    baselines = {m: np.mean(proba_test[m], axis=0) for m in C.MODELS_MAIN
                 if m in proba_test and m != BEST}
    table12, ci_best = E.table_significance(split.yte, p_ens, baselines, groups=split.gte)
    save_table(table12, "table12_significativite",
               "Test de McNemar et IC95 bootstrap par patient.", "tab:stats", "%s")

    save_table(E.table_literature(), "table13_etat_de_l_art",
               "Comparaison aux travaux inter-patient de reference.", "tab:sota")

    per_patient = E.table_per_patient(split.yte, p_ens.argmax(1), split.gte)
    save_table(per_patient, "table14_par_patient", "Performances par enregistrement de DS2.", "tab:patients")
    P.fig_per_patient(per_patient)

    # ------------------------------------------------------------- figures
    print("\n[5/9] Figures")
    if histories:
        P.fig_learning_curves(histories)
    P.fig_confusion(split.yte, p_ens.argmax(1), f"{BEST} — DS2", "fig04_confusion.png")
    P.fig_confusion(split.yte, pred_clin, "Point de fonctionnement clinique", "fig04b_confusion_clinique.png")
    P.fig_roc_pr(split.yte, p_ens)
    P.fig_model_comparison(runs[runs.Model.isin(C.MODELS_MAIN)])
    dc = decision_curve((split.yte == C.IDX_V).astype(int), p_ens[:, C.IDX_V])
    save_table(dc, "table15_decision_curve", "Benefice net (analyse de la courbe de decision).", "tab:dca")
    P.fig_decision_curve(dc)

    comp = runs[runs.Model.isin(C.MODELS_MAIN)].groupby("Model").agg(
        Params=("Params", "first"), Latence_ms=("Latence_ms", "mean"),
        F1_macro=("F1_macro", "mean")).reset_index()
    comp["Taille_Mo"] = comp.Params * 4 / 1e6
    comp["Battements_par_seconde"] = 1000 / comp.Latence_ms.clip(lower=1e-6)
    save_table(comp, "table16_complexite", "Complexite et latence.", "tab:complexity")
    P.fig_complexity(comp)

    # ---------------------------------------------- interpretabilite
    print("\n[6/9] Interpretabilite")
    try:
        import tensorflow as tf
        from src.models import model_inputs

        def integrated_gradients(model, xb, rb, target, steps=64):
            xt = tf.convert_to_tensor(xb[None], tf.float32)
            base = tf.zeros_like(xt)
            alphas = tf.linspace(0.0, 1.0, steps + 1)[:, None, None]
            interp = base + alphas * (xt - base)
            feed = {"signal": interp, "rr": tf.repeat(tf.convert_to_tensor(rb[None], tf.float32),
                                                      steps + 1, axis=0)} \
                if len(model.inputs) == 2 else interp
            with tf.GradientTape() as tape:
                tape.watch(interp)
                out = model(feed, training=False)[:, target]
            grads = tape.gradient(out, interp)
            attr = (xt - base)[0] * tf.reduce_mean(grads[:-1], axis=0)
            return attr.numpy()[:, 1]

        rng = np.random.default_rng(C.SEEDS[0])
        pred = p_ens.argmax(1)
        picks = []
        for i in range(C.N_CLASSES):
            cand = np.flatnonzero((split.yte == i) & (pred == i))
            if len(cand) == 0:
                cand = np.flatnonzero(split.yte == i)
            if len(cand):
                k = int(rng.choice(cand))
                picks.append((k, integrated_gradients(keep_models[BEST], split.Xte[k], split.Rte[k], i), i))
        if picks:
            P.fig_interpretability(split.Xte, split.yte, pred, picks)
    except Exception as exc:
        print("  interpretabilite ignoree :", exc)

    # ------------------------------------------------- validation externe
    external_summary = None
    if args.external:
        print("\n[7/9] Validation externe (sans reentrainement)")
        try:
            ext_dir = Path(args.external)
            ext_records = sorted({p.stem for p in ext_dir.glob("*.dat")})
            Xe, RRe, ye, ge, te, dure = extract_records(ext_dir, ext_records,
                                                        cache_name=f"external_{ext_dir.name}")
            from src.preprocessing import stack_context_by_record
            Xec = stack_context_by_record(Xe, ge)
            Ree = split.rr_scaler.transform(RRe)
            pe = predict_proba(keep_models[BEST], Xec, Ree)
            ge_metrics, ct_ext = E.global_metrics(ye, pe)
            ext_table = pd.DataFrame([{"Base": ext_dir.name, "Battements": len(ye),
                                       **{k: round(float(v), 4) for k, v in ge_metrics.items()
                                          if isinstance(v, (int, float, np.floating))}}])
            save_table(ext_table, "table17_validation_externe",
                       "Validation externe sans reentrainement.", "tab:external")
            save_table(ct_ext, "table18_validation_externe_par_classe",
                       "Metriques cliniques sur la base externe.", "tab:external_class")
            P.fig_confusion(ye, pe.argmax(1), f"Validation externe — {ext_dir.name}",
                            "fig15_confusion_externe.png")
            external_summary = ext_table.to_dict("records")[0]
        except Exception as exc:
            print("  validation externe ignoree :", exc)

    # ------------------------------------------------------------- resume
    print("\n[8/9] Resume")
    final, _ = E.global_metrics(split.yte, p_ens)
    summary = {
        "protocole": {
            "split": "inter-patient DS1/DS2 (de Chazal)",
            "records_stimules_exclus": C.PACED_RECORDS,
            "patients_train": sorted(set(split.gtr.tolist())),
            "patients_validation": sorted(set(split.gva.tolist())),
            "patients_test": sorted(set(split.gte.tolist())),
            "adaptation_patient": use_adapt,
            "secondes_adaptation_par_patient": C.ADAPT_SECONDS if use_adapt else 0,
            "battements_test": int(len(split.yte)),
        },
        "modele": BEST, "seeds": C.SEEDS,
        "metriques_test": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in final.items()},
        "f1_macro_ic95_par_patient": ci_best,
        "calibration": calib_info,
        "temperature": T,
        "seuils_cliniques": {C.CLASS_NAMES[k]: float(v) for k, v in thresholds.items() if v <= 1},
        "option_de_rejet_99pct": deferral,
        "alarmes_par_heure_par_battement": float(alarms[alarms.Enregistrement == "GLOBAL"].FA_par_heure.iloc[0]),
        "alarmes_par_heure_declenchement": float(alarms_db[alarms_db.Enregistrement == "GLOBAL"].FA_par_heure.iloc[0]),
        "sensibilite_alarme": float(alarms_db[alarms_db.Enregistrement == "GLOBAL"].Sensibilite_alarme.iloc[0]),
        "vpp_alarme": float(alarms_db[alarms_db.Enregistrement == "GLOBAL"].VPP_alarme.iloc[0]),
        "validation_externe": external_summary,
        "duree_totale_min": round((time.time() - t_start) / 60, 1),
    }
    (C.OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

    print("\n[9/9] TERMINE")
    for k in ["Accuracy", "Balanced_Acc", "F1_macro", "MCC_macro", "Kappa", "AUC_macro",
              "Se_S", "PP_S", "Se_V", "PP_V", "Brier", "ECE"]:
        print(f"  {k:<14}: {final[k]:.4f}")
    print(f"  F1 macro IC95 : [{ci_best['ci95_low']:.4f}, {ci_best['ci95_high']:.4f}]")
    print(f"  FA/h par battement       : {summary['alarmes_par_heure_par_battement']:.2f}")
    print(f"  FA/h regle 3-sur-10      : {summary['alarmes_par_heure_declenchement']:.2f}"
          f"  (Se_alarme {summary['sensibilite_alarme']:.3f} | VPP_alarme {summary['vpp_alarme']:.3f})")
    print(f"  Duree : {summary['duree_totale_min']} min")
    print(f"\n  Tables  : {C.TAB_DIR}\n  Figures : {C.FIG_DIR}\n  Resume  : {C.OUT_DIR/'summary.json'}")


if __name__ == "__main__":
    main()
