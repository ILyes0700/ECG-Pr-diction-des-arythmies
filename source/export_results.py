"""Rassemble TOUS les resultats en un seul livrable.

    python export_results.py                 # depuis le dossier du projet
    python export_results.py --out /content/drive/MyDrive

Produit :
  results/RESULTS.md          synthese lisible (chiffres cles + toutes les tables)
  results/resultats.xlsx      un onglet par table
  results/tables_latex.tex    toutes les tables LaTeX concatenees
  ecg_resultats_<date>.zip    tables + figures + summary.json + RESULTS.md
"""
from __future__ import annotations
import argparse, json, shutil, zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd

HEADLINE = ["Accuracy", "Balanced_Acc", "F1_macro", "MCC_macro", "Kappa", "AUC_macro",
            "Se_S", "PP_S", "Se_V", "PP_V", "Brier", "ECE"]
TITLES = {
    "table01_dataset": "Table I — Composition de la base",
    "table02_comparaison": "Table II — Comparaison des modeles (DS2, inter-patient)",
    "table03_metriques_cliniques": "Table III — Metriques diagnostiques par classe",
    "table04_prevalence": "Table IV — VPP/VPN ajustes a la prevalence",
    "table05_ec57": "Table V — Statistiques AAMI EC57 (gross / average)",
    "table06_alarmes": "Table VI — Charge d'alarme par patient",
    "table07_episodes": "Table VII — Detection par episode et latence",
    "table08_calibration": "Table VIII — Calibration probabiliste",
    "table09_point_de_fonctionnement": "Table IX — Points de fonctionnement cliniques",
    "table10_couverture_risque": "Table X — Couverture / risque (option de rejet)",
    "table11_ablation": "Table XI — Etude d'ablation",
    "table12_significativite": "Table XII — Significativite statistique",
    "table13_etat_de_l_art": "Table XIII — Comparaison a l'etat de l'art",
    "table14_par_patient": "Table XIV — Resultats par enregistrement",
    "table15_decision_curve": "Table XV — Benefice net (decision curve)",
    "table16_complexite": "Table XVI — Complexite et latence",
    "table17_validation_externe": "Table XVII — Validation externe",
    "table18_validation_externe_par_classe": "Table XVIII — Validation externe par classe",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=".", help="ou deposer le zip (ex: /content/drive/MyDrive)")
    args = ap.parse_args()

    res = Path(args.results)
    tab, fig = res / "tables", res / "figures"
    if not tab.exists():
        raise SystemExit(f"Dossier introuvable : {tab}. Lancez d'abord run_all.py.")

    csvs = sorted(tab.glob("*.csv"))
    frames = {p.stem: pd.read_csv(p) for p in csvs}
    summary = {}
    if (res / "summary.json").exists():
        summary = json.loads((res / "summary.json").read_text(encoding="utf-8"))

    # ------------------------------------------------------------ RESULTS.md
    lines = ["# Resultats de l'experience", "",
             f"_Genere le {datetime.now():%Y-%m-%d %H:%M}_", ""]
    proto = summary.get("protocole", {})
    if proto:
        lines += ["## Protocole", "",
                  f"- Split : {proto.get('split')}",
                  f"- Enregistrements stimules exclus : {proto.get('records_stimules_exclus')}",
                  f"- Patients test : {len(proto.get('patients_test', []))} "
                  f"({', '.join(proto.get('patients_test', [])[:8])}...)",
                  f"- Adaptation patient : {proto.get('adaptation_patient')} "
                  f"({proto.get('secondes_adaptation_par_patient')} s/patient, exclues du test)",
                  f"- Battements de test : {proto.get('battements_test'):,}" if proto.get("battements_test") else "",
                  ""]
    m = summary.get("metriques_test", {})
    if m:
        lines += ["## Chiffres cles", "", "| Metrique | Valeur |", "|---|---|"]
        for k in HEADLINE:
            if k in m:
                lines.append(f"| {k} | {m[k]:.4f} |")
        ci = summary.get("f1_macro_ic95_par_patient", {})
        if ci:
            lines.append(f"| F1 macro IC95 (par patient) | [{ci['ci95_low']:.4f}, {ci['ci95_high']:.4f}] |")
        if "alarmes_par_heure" in summary:
            lines.append(f"| Fausses alarmes V / heure | {summary['alarmes_par_heure']:.2f} |")
        if "temperature" in summary:
            lines.append(f"| Temperature de calibration | {summary['temperature']:.3f} |")
        lines.append("")

    def to_md(df):
        d = df.round(4)
        try:
            return d.to_markdown(index=False)          # necessite tabulate
        except Exception:                               # repli sans dependance
            head = "| " + " | ".join(map(str, d.columns)) + " |"
            sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
            rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
            return "\n".join([head, sep] + rows)

    lines += ["## Tables", ""]
    for stem, df in frames.items():
        lines += [f"### {TITLES.get(stem, stem)}", "", to_md(df), ""]

    figs = sorted(fig.glob("*.png"))
    if figs:
        lines += ["## Figures", ""] + [f"- `figures/{f.name}`" for f in figs] + [""]

    (res / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("  ->", res / "RESULTS.md")

    # ------------------------------------------------------------- Excel
    try:
        with pd.ExcelWriter(res / "resultats.xlsx", engine="xlsxwriter") as xl:
            for stem, df in frames.items():
                df.to_excel(xl, sheet_name=stem[:31], index=False)
        print("  ->", res / "resultats.xlsx")
    except Exception as exc:
        print("  [excel ignore]", exc)

    # -------------------------------------------------------- LaTeX unique
    texs = sorted(tab.glob("*.tex"))
    if texs:
        (res / "tables_latex.tex").write_text(
            "\n\n".join(p.read_text(encoding="utf-8") for p in texs), encoding="utf-8")
        print("  ->", res / "tables_latex.tex")

    # ------------------------------------------------------------- archive
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    zip_path = Path(args.out) / f"ecg_resultats_{stamp}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in list(tab.glob("*")) + list(fig.glob("*.png")):
            z.write(p, Path(p.parent.name) / p.name)
        for extra in ["RESULTS.md", "resultats.xlsx", "tables_latex.tex", "summary.json"]:
            if (res / extra).exists():
                z.write(res / extra, extra)
    size = zip_path.stat().st_size / 1e6
    print(f"\nArchive : {zip_path}  ({size:.1f} Mo)")
    print(f"Contenu : {len(csvs)} tables CSV, {len(texs)} tables LaTeX, {len(figs)} figures")
    return zip_path


if __name__ == "__main__":
    main()
