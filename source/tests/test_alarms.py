import sys, numpy as np
sys.path.insert(0, ".")
from src import config as C
from src.clinical_metrics import _trigger_events, alarm_burden_debounced

# --- regle k parmi n ------------------------------------------------------- #
t = np.arange(100) * 0.8                       # 75 bpm
flags = np.zeros(100, bool); flags[20:23] = True      # une salve de 3
ev = _trigger_events(flags, t, (3, 10), 60.0)
assert len(ev) == 1, ev
flags2 = np.zeros(100, bool); flags2[[20, 40, 60]] = True   # 3 isoles => aucune alarme
assert len(_trigger_events(flags2, t, (3, 10), 60.0)) == 0
print("regle 3-sur-10 OK")

# --- periode refractaire --------------------------------------------------- #
long_t = np.arange(2000) * 0.8                 # ~27 min
f = np.zeros(2000, bool)
for s in range(0, 2000, 25):                   # une salve toutes les 20 s
    f[s:s+3] = True
ev = _trigger_events(f, long_t, (3, 10), 60.0)
gaps = np.diff(ev)
assert (gaps >= 60.0).all(), gaps
print(f"periode refractaire OK  ({len(ev)} alarmes au lieu de {2000//25})")

# --- charge d'alarmes ------------------------------------------------------ #
n = 4000
rng = np.random.default_rng(0)
times = np.arange(n) * 0.8
yt = np.zeros(n, int)
for s in range(200, n, 700):
    yt[s:s+5] = C.IDX_V
yp = yt.copy()
flip = rng.choice(n, 40, replace=False)         # bruit isole : ne doit pas alarmer
yp[flip] = C.IDX_V
groups = np.full(n, "200")
df = alarm_burden_debounced(yt, yp, groups, times, {"200": n * 0.8}, C.CLASS_NAMES)
g = df[df.Enregistrement == "GLOBAL"].iloc[0]
assert g.Sensibilite_alarme == 1.0, g.Sensibilite_alarme
assert g.FA_par_heure < 2.0, g.FA_par_heure
print(f"charge d'alarmes OK  Se={g.Sensibilite_alarme:.3f} VPP={g.VPP_alarme:.3f} "
      f"FA/h={g.FA_par_heure:.2f}")

print("\nTESTS ALARMES : OK")
