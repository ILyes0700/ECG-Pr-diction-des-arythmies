"""Configuration centrale de l'experience. Un seul endroit a modifier."""
from pathlib import Path
import numpy as np

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("data")                 # surchargé par run_all.py / le notebook
MITDB_DIR = DATA_ROOT / "mitdb"
INCART_DIR = DATA_ROOT / "incartdb"
SVDB_DIR = DATA_ROOT / "svdb"

import os

# ECG_RESULTS_DIR permet de stocker les resultats sur Google Drive : rien n'est perdu
# si la session Colab est deconnectee.
OUT_DIR = Path(os.environ.get("ECG_RESULTS_DIR", ROOT / "results"))
CACHE_DIR = OUT_DIR / "cache"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"
MOD_DIR = OUT_DIR / "models"

# --------------------------------------------------------------------------- #
# Signal
# --------------------------------------------------------------------------- #
FS = 360                    # Hz (MIT-BIH). Les autres bases sont rééchantillonnées à FS.
BP_LOW, BP_HIGH = 0.5, 40.0  # bande diagnostique AHA
PRE, POST = 90, 110          # 250 ms avant R / 306 ms après R
RESAMPLE_LEN = 128           # longueur d'un battement après rééchantillonnage
N_CHANNELS = 3               # battement precedent | courant | suivant  (contexte rythmique)
RR_DIM = 6                   # pre, post, local10, ratio_local, ratio_global, post/pre

# --------------------------------------------------------------------------- #
# Protocole AAMI EC57
# --------------------------------------------------------------------------- #
# La classe Q (inclassable / stimule) est exclue : apres retrait des enregistrements
# stimules impose par l'AAMI EC57, elle ne compte plus que 15 battements sur 100 705,
# ce qui produit un poids de classe aberrant et fait diverger l'apprentissage.
# L'evaluation porte sur les quatre classes cliniquement pertinentes N, S, V, F,
# comme dans la litterature inter-patient de reference (de Chazal 2004, Mar 2011).
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}
CLASS_NAMES = np.array(["N", "S", "V", "F"])
N_CLASSES = len(CLASS_NAMES)
CLASS_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_N, IDX_S, IDX_V, IDX_F = 0, 1, 2, 3

PACED_RECORDS = ["102", "104", "107", "217"]          # exclus (convention EC57)
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
VAL_RECORDS = ["101", "114", "201", "209", "223"]      # validation, retirés de DS1

# --------------------------------------------------------------------------- #
# Adaptation patient (protocole AAMI : les 5 premieres minutes sont "connues")
# --------------------------------------------------------------------------- #
ADAPT_SECONDS = 300          # 5 min de chaque enregistrement DS2
ADAPT_EPOCHS = 8
ADAPT_LR = 1e-4

# --------------------------------------------------------------------------- #
# Entrainement
# --------------------------------------------------------------------------- #
SEEDS = [42, 1337, 2024, 7, 99]
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
BOOTSTRAP_N = 1000
MC_DROPOUT_PASSES = 30

# Cibles cliniques imposees au point de fonctionnement (choisi sur la VALIDATION)
# Ponderation des classes. L'inverse pur de la frequence (puissance 1.0) donne un
# rapport N:F de 1:90 sur MIT-BIH : la perte focale sature et le reseau predit les
# classes rares partout (Accuracy 0.10, VPP_S 0.06). La puissance 0.5, normalisee a
# moyenne 1 puis plafonnee, ramene ce rapport a 1:13 et restaure la calibration.
CLASS_WEIGHT_POWER = 0.5
MAX_CLASS_WEIGHT = 10.0

TARGET_SE_V = 0.95
TARGET_SE_S = 0.85

# Cout clinique d'une erreur (rater une extrasystole ventriculaire est le plus grave)
COST_MATRIX = np.array([
    #  pred   N    S    V    F
    [0.0, 1.0, 2.0, 1.0],    # vrai N
    [5.0, 0.0, 3.0, 3.0],    # vrai S
    [10.0, 6.0, 0.0, 3.0],   # vrai V (rater un battement ventriculaire = le plus grave)
    [6.0, 4.0, 2.0, 0.0],    # vrai F
], dtype=float)

# Prevalences realistes pour l'ajustement bayesien des VPP/VPN
PREVALENCE_GRID = [0.01, 0.05, 0.10, 0.20]

MODELS_MAIN = ["CNN1D", "BiLSTM", "CNN_BiLSTM", "ResNet1D", "PROPOSED"]
MODELS_ABLATION = ["ABL_noRR", "ABL_noCTX", "ABL_noATT", "ABL_noSE", "ABL_CE", "ABL_noAUG"]

QUICK_RUN = False


def apply_quick_run():
    """Mode mise au point : 1 seed, peu d'epoques."""
    global EPOCHS, PATIENCE, BOOTSTRAP_N, SEEDS, MC_DROPOUT_PASSES, ADAPT_EPOCHS
    EPOCHS, PATIENCE, BOOTSTRAP_N = 12, 6, 200
    SEEDS = [42]
    MC_DROPOUT_PASSES = 10
    ADAPT_EPOCHS = 4


def ensure_dirs():
    for d in (CACHE_DIR, TAB_DIR, FIG_DIR, MOD_DIR):
        d.mkdir(parents=True, exist_ok=True)
