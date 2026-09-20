# Inter-Patient ECG Arrhythmia Classification — pipeline clinique reproductible

Classification des battements ECG en 5 classes AAMI (N, S, V, F, Q) sur MIT-BIH, sous protocole
**inter-patient DS1/DS2**, avec une couche complete de **metriques cliniques**, la calibration
probabiliste, l'interpretabilite et une option de rejet.

## Pourquoi ce depot

La plupart des travaux publies annoncent > 99 % d'exactitude sur MIT-BIH avec un decoupage
intra-patient : des battements du meme patient se retrouvent en apprentissage et en test. Ici :

- **aucun patient partage** entre train, validation et test (verifie par assertions) ;
- **aucun reequilibrage** sur validation/test, aucune statistique apprise sur le test ;
- seuils de decision, temperature de calibration et arret anticipe fixes **sur la validation** ;
- resultats donnes avec **IC95 bootstrap par patient**, **test de McNemar** et **5 graines**.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Donnees

```bash
mkdir -p data/mitdb
wget -r -N -c -np -nH --cut-dirs=4 -P data/mitdb https://physionet.org/files/mitdb/1.0.0/
# validation externe (optionnelle)
wget -r -N -c -np -nH --cut-dirs=4 -P data/incartdb https://physionet.org/files/incartdb/1.0.0/
```

## Reproduction complete

```bash
pytest tests/ -q                                    # verifie la logique d'evaluation
python run_all.py --data data/mitdb --quick         # mise au point (~30 min GPU)
python run_all.py --data data/mitdb                 # run final 5 graines (~5 h GPU)
python run_all.py --data data/mitdb --external data/incartdb
```

Sorties : `results/tables/*.csv` et `*.tex`, `results/figures/*.png` (330 dpi),
`results/models/*.keras`, `results/summary.json`.

Version Colab interactive : `ECG_CLINICAL_PIPELINE.ipynb`.
**PC peu puissant ? Suivez `GUIDE_COLAB.md` et le notebook `RUN_COLAB.ipynb`** (entrainement sur
Colab GPU, resultats ecrits sur Google Drive, reprise automatique apres deconnexion).

Extraction de tous les resultats en un livrable unique :

```bash
python export_results.py --results results --out .
# -> RESULTS.md, resultats.xlsx, tables_latex.tex, ecg_resultats_<date>.zip
```

## Protocole

| Ensemble | Enregistrements | Role |
|---|---|---|
| Train | DS1 moins 5 patients (17) | apprentissage |
| Validation | 101, 114, 201, 209, 223 | arret anticipe, seuils, temperature |
| Test | DS2 (22 patients) | evaluation unique et finale |
| Adaptation (option AAMI) | 5 premieres minutes de chaque enregistrement DS2 | reglage fin, **exclues du test** |

Enregistrements stimules 102, 104, 107, 217 exclus (convention EC57).

## Methode

1. Passe-bande 0,5-40 Hz zero-phase + suppression de derive par filtres medians.
2. Fenetre asymetrique autour de R (-250 ms / +306 ms), rééchantillonnage a 128 points,
   z-score par battement.
3. Entree a 3 canaux : battement precedent | courant | suivant (contexte rythmique).
4. 6 descripteurs RR : pre, post, moyenne locale (10), pre/local, pre/global, post/pre.
5. Reseau **MS-Res-SE-BiGRU-Attention** avec fusion tardive des RR.
6. Focal loss ponderee (gamma = 2) + augmentation realiste (bruit, gain, decalage, warping).
7. Ensemble de 5 graines, temperature scaling, seuils a sensibilite cible.

## Metriques produites

Se, Sp, VPP, VPN, F1, FPR, FNR, **LR+**, **LR-**, **DOR**, **Youden J**, **MCC**, **NND**,
**CUI+/CUI-**, **VPP/VPN ajustes a la prevalence**, **EC57 gross et average**,
**fausses alarmes par heure**, **battements manques par heure**, **sensibilite par episode et
latence**, **benefice net (decision curve)**, **cout clinique pondere**, **Brier / ECE / MCE**,
**courbe couverture-risque et taux de renvoi a 99 % de fiabilite**.

## Structure

```
src/config.py            protocole et hyperparametres
src/preprocessing.py     filtrage, segmentation, RR, augmentation
src/datasets.py          extraction, cache, split verrouille
src/models.py            architecture proposee + 4 baselines + ablations
src/losses.py            focal, CE ponderee, cout clinique
src/clinical_metrics.py  metriques sanitaires (numpy pur, testees)
src/calibration.py       temperature scaling
src/uncertainty.py       MC-dropout, option de rejet
src/evaluate.py          courbes, tables, tests statistiques
src/train.py             boucle multi-graines, adaptation patient, TTA
src/plots.py             15 figures publication
run_all.py               pipeline complet
tests/                   tests unitaires de la logique d'evaluation
paper/                   squelette IEEE, bibliographie, cover letter
```

## Licence et citation

MIT. Voir `CITATION.cff`.
