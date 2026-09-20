# Guide d'execution sur Google Colab (PC faible)

Tout se passe sur Colab. Votre ordinateur ne sert qu'a ouvrir le navigateur et a recevoir le
zip final des resultats.

---

## Resume en 9 etapes

| # | Action | Duree | Ou |
|---|---|---|---|
| 0 | Activer le GPU T4 | 10 s | Colab |
| 1 | Monter Google Drive | 30 s | Colab |
| 2 | Installer le projet (zip) | 1 min | Colab |
| 3 | Installer les dependances | 2 min | Colab |
| 4 | Detecter vos donnees Drive (ou telecharger) + verification | 1-3 min | Colab |
| 5 | Lancer les tests unitaires | 30 s | Colab |
| 6 | Test rapide `--quick` | 25-40 min | Colab |
| 7 | Run final (5 graines) | 4-6 h | Colab |
| 8 | Extraire les resultats | 1 min | Colab |
| 9 | Telecharger le zip | 1 min | Votre PC |

Le notebook **`RUN_COLAB.ipynb`** contient exactement ces etapes, une cellule par etape.

---

## 0. Preparer Colab

1. Ouvrir <https://colab.research.google.com>.
2. **Fichier → Importer un notebook** → choisir `RUN_COLAB.ipynb`.
3. **Execution → Modifier le type d'execution → T4 GPU → Enregistrer.**

Sans GPU, l'entrainement passe de 5 h a plus de 40 h : ne sautez pas cette etape.

## 1. Deposer le projet sur le Drive

Depuis votre PC, copiez `ecg_arrhythmia_paper.zip` dans **Mon Drive** (racine).
Le notebook le trouvera tout seul. Sinon il vous proposera de le televerser.

## 2. Pourquoi les resultats vont sur le Drive

La cellule 1 definit `ECG_RESULTS_DIR = /content/drive/MyDrive/ecg_results`.
Tables, figures, modeles et probabilites de test y sont ecrits au fur et a mesure.
Si Colab vous deconnecte, **rien n'est perdu**.

## 2 bis. Utiliser VOS donnees deja presentes sur le Drive

La cellule 4 detecte automatiquement le dossier contenant les `.dat` sous `MyDrive`.
Deux cas particuliers :

- **Dossier partage par lien** (« Partages avec moi ») : Drive ne le monte pas dans `MyDrive`.
  Dans Google Drive, clic droit sur le dossier → **Organiser → Ajouter un raccourci → Mon Drive**,
  puis relancez la cellule 1 et la cellule 4.
- **Chemin impose** : renseignez `MES_DONNEES_DRIVE = "/content/drive/MyDrive/<votre dossier>"`.

La verification est faite par `check_data.py` :

```bash
python check_data.py --data "/content/drive/MyDrive/<votre dossier>"
```

Il controle : sous-dossier reel contenant les `.dat`, triplets `.dat/.hea/.atr` complets,
presence des 44 enregistrements DS1/DS2, lecture wfdb, frequence, derivation et comptage AAMI.
Il refuse de valider un jeu incomplet, ce qui evite un entrainement rate au bout de 4 heures.

## 3. Toujours faire le test rapide avant le run final

```bash
python run_all.py --data /content/mitdb --quick
```

Il utilise 1 graine et 12 epoques. S'il se termine par « TERMINE » et affiche des metriques,
la chaine complete fonctionne. Ses chiffres ne sont pas publiables.

## 4. Le run final et la reprise automatique

```bash
python run_all.py --data /content/mitdb
```

Chaque couple (modele, graine) est sauvegarde dans `ecg_results/proba/`.
En cas de deconnexion : refaites les cellules 1 a 3, puis relancez la meme commande.
Vous verrez `[reprise] PROPOSED | seed 42 deja calcule` : le travail deja fait n'est pas refait.

Si votre session ne tient pas 5 h, lancez par lots :

```bash
python run_all.py --data /content/mitdb --models PROPOSED
python run_all.py --data /content/mitdb --models CNN1D,BiLSTM
python run_all.py --data /content/mitdb --models CNN_BiLSTM,ResNet1D
python run_all.py --data /content/mitdb --models ABL_noRR,ABL_noCTX,ABL_noATT,ABL_noSE,ABL_CE,ABL_noAUG
python run_all.py --data /content/mitdb        # passe final : toutes les tables et figures
```

Le dernier appel est obligatoire : c'est lui qui agrege tout.

## 5. Extraire les resultats

```bash
python export_results.py --results "$ECG_RESULTS_DIR" --out /content/drive/MyDrive
```

Produit :

- `RESULTS.md` — synthese lisible avec les chiffres cles et toutes les tables ;
- `resultats.xlsx` — un onglet par table ;
- `tables_latex.tex` — toutes les tables LaTeX concatenees, a coller dans l'article ;
- `ecg_resultats_<date>.zip` — l'archive complete (tables + figures + summary.json).

## 6. Ce que vous recuperez

**18 tables** : composition de la base, comparaison des modeles, metriques diagnostiques par
classe (Se, Sp, VPP, VPN, LR+, LR-, DOR, Youden, MCC, NND), VPP/VPN ajustes a la prevalence,
EC57 gross/average, alarmes par heure, episodes et latence, calibration, points de
fonctionnement, couverture-risque, ablation, McNemar, etat de l'art, par patient, benefice net,
complexite, validation externe.

**15 figures** 330 dpi : morphologies par classe, distribution, courbes d'apprentissage,
matrices de confusion (argmax et point clinique), ROC/PR, comparaison, calibration,
benefice net, couverture-risque, alarmes, par patient, ablation, interpretabilite, complexite.

---

## Tableau des erreurs et solutions

| Message / symptome | Cause | Solution |
|---|---|---|
| `assert ... GPU non actif` | runtime CPU | Execution → Modifier le type d'execution → T4 GPU |
| Bouton `RESTART SESSION` apres `pip` | numpy/pandas recharges | Cliquer, puis refaire les cellules 1, 2, 3 |
| `Aucun fichier .dat trouve` | telechargement coupe | Relancer la cellule 4 : `wget -c` reprend |
| Donnees sur Drive non vues | dossier en « Partages avec moi » | Drive : clic droit → Organiser → Ajouter un raccourci → Mon Drive |
| `enregistrements requis absents` | jeu incomplet | Completer le dossier ou laisser telecharger depuis PhysioNet |
| `FileNotFoundError: src/config.py` | mauvais repertoire courant | Refaire la cellule 2 (contient `%cd`) |
| `ModuleNotFoundError: wfdb` | session redemarree | Refaire la cellule 3 |
| `ResourceExhaustedError` (OOM GPU) | batch trop grand | `BATCH_SIZE = 128` dans `src/config.py` |
| `AttributeError: module keras.optimizers has no attribute AdamW` | TF < 2.11 | `!pip install -U "tensorflow>=2.16"` puis redemarrer |
| Session deconnectee a 3 h d'entrainement | limite Colab gratuit | Cellules 1-3 puis relancer : reprise automatique |
| `KeyError: 'PROPOSED'` a l'export | run interrompu avant la fin | Relancer le run jusqu'a « TERMINE » |
| Le Drive ne contient rien | `ECG_RESULTS_DIR` non defini | Refaire la cellule 1 avant tout `run_all.py` |
| Un test unitaire echoue | fichier source modifie | Reinstaller depuis le zip |
| Entrainement tres lent | GPU non utilise par TF | Verifier `tf.config.list_physical_devices("GPU")` |
| `UnicodeDecodeError` a l'export | fichier CSV corrompu | Supprimer le CSV fautif et relancer le run final |

---

## Reperes de duree (GPU T4)

| Etape | Duree |
|---|---|
| Extraction + cache des battements | 8-12 min (une seule fois) |
| PROPOSED, 1 graine, 60 epoques | 35-50 min |
| Une baseline, 1 graine | 10-20 min |
| Run complet : 11 modeles x 5 graines | 4-6 h avec reprise |
| Tables + figures + bootstrap | 6-10 min |

## Trois regles a ne jamais enfreindre

1. Ne modifiez pas `src/datasets.py` : ses assertions garantissent l'absence de fuite entre
   patients, donc la validite de vos resultats.
2. Ne publiez jamais les chiffres du mode `--quick`.
3. Executez toujours le run final d'un seul tenant logique (reprises autorisees), puis exportez.
   Ne melangez pas des resultats issus de configurations differentes.
