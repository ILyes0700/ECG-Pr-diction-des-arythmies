"""Verifie qu'un dossier contient bien la base MIT-BIH exploitable par le pipeline.

    python check_data.py --data /content/drive/MyDrive/mit-bih-arrhythmia-database-1.0.0

Controle : localisation reelle des fichiers, triplets .dat/.hea/.atr complets,
presence des 44 enregistrements requis par DS1/DS2, lisibilite via wfdb,
frequence d'echantillonnage, derivation, et comptage des classes AAMI.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from src import config as C


def locate_data(root: Path) -> Path:
    """Trouve le dossier contenant reellement les .dat (souvent un sous-dossier)."""
    root = Path(root)
    if any(root.glob("*.dat")):
        return root
    candidates = sorted({p.parent for p in root.rglob("*.dat")})
    if not candidates:
        raise SystemExit(f"Aucun fichier .dat trouve sous {root}")
    best = max(candidates, key=lambda d: len(list(d.glob("*.dat"))))
    print(f"  fichiers .dat trouves dans un sous-dossier : {best}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    data = locate_data(Path(args.data))
    stems = sorted({p.stem for p in data.glob("*.dat")})
    print(f"\nDossier    : {data}")
    print(f"Enregistrements .dat : {len(stems)}")

    problems = []

    # 1) triplets complets
    incomplete = [s for s in stems
                  if not (data / f"{s}.hea").exists() or not (data / f"{s}.atr").exists()]
    if incomplete:
        problems.append(f"fichiers .hea ou .atr manquants pour : {incomplete}")

    # 2) enregistrements requis par le protocole
    required = set(C.DS1 + C.DS2)
    missing = sorted(required - set(stems))
    if missing:
        problems.append(f"enregistrements requis absents (DS1/DS2) : {missing}")
    paced_present = sorted(set(stems) & set(C.PACED_RECORDS))

    # 3) lecture reelle d'un echantillon
    fs_values, lead_names, total_beats, class_counts = set(), set(), 0, {}
    try:
        import wfdb
        probe = [s for s in (C.DS1[:2] + C.DS2[:2]) if s in stems] or stems[:3]
        for rec in probe:
            r = wfdb.rdrecord(str(data / rec))
            a = wfdb.rdann(str(data / rec), "atr")
            fs_values.add(float(r.fs))
            lead_names.add(r.sig_name[0] if r.sig_name else "?")
            total_beats += sum(1 for s in a.symbol if s in C.AAMI_MAP)
            for s in a.symbol:
                if s in C.AAMI_MAP:
                    cls = C.AAMI_MAP[s]
                    class_counts[cls] = class_counts.get(cls, 0) + 1
        print(f"Lecture wfdb        : OK sur {len(probe)} enregistrements testes")
        print(f"Frequences          : {sorted(fs_values)} Hz")
        print(f"Derivation canal 0  : {sorted(lead_names)}")
        print(f"Battements AAMI (echantillon) : {total_beats:,} -> {class_counts}")
        if fs_values and fs_values != {360.0}:
            print("  note : frequence differente de 360 Hz, le pipeline rééchantillonne "
                  "automatiquement la fenetre.")
    except ImportError:
        print("Lecture wfdb        : ignoree (module wfdb absent ici)")
    except Exception as exc:
        problems.append(f"lecture wfdb impossible : {exc}")

    print(f"\nEnregistrements stimules presents (seront exclus) : {paced_present or 'aucun'}")

    if problems:
        print("\nPROBLEMES DETECTES")
        for p in problems:
            print("  -", p)
        print("\nCorrigez avant de lancer run_all.py.")
        raise SystemExit(1)

    print("\nDONNEES VALIDES — utilisables telles quelles.")
    print(f"Commande a lancer :\n  python run_all.py --data \"{data}\" --quick")
    return data


if __name__ == "__main__":
    main()
