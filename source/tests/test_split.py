import sys, numpy as np
sys.path.insert(0, ".")
from src import config as C
from src.datasets import build_split, test_durations_after_adaptation

rng = np.random.default_rng(0)
recs = C.DS1 + C.DS2
X, RR, y, g, t = [], [], [], [], []
dur = {}
for r in recs:
    n = 600
    X.append(rng.normal(size=(n, C.RESAMPLE_LEN)).astype(np.float32))
    RR.append(rng.normal(size=(n, C.RR_DIM)).astype(np.float32))
    y.append(rng.integers(0, C.N_CLASSES, n))
    g.append(np.array([r]*n))
    t.append(np.sort(rng.uniform(0, 1800, n)))     # 30 min par enregistrement
    dur[r] = 1800.0
X = np.vstack(X); RR = np.vstack(RR); y = np.concatenate(y); g = np.concatenate(g); t = np.concatenate(t)

s = build_split(X, RR, y, g, t, dur, use_adaptation=True, verbose=False)
print(s.summary())

# 1. patients disjoints
assert not (set(s.gtr) & set(s.gva)) and not (set(s.gtr) & set(s.gte)) and not (set(s.gva) & set(s.gte))
# 2. le test ne contient QUE des patients DS2
assert set(s.gte) <= set(C.DS2) and set(s.gtr) <= set(C.DS1) and set(s.gva) == set(C.VAL_RECORDS)
# 3. adaptation et test disjoints (aucun battement commun) et adaptation issue de DS2 seulement
assert set(s.gad) <= set(C.DS2)
assert len(s.yad) + len(s.yte) == int(np.isin(g, C.DS2).sum())
# 4. proportion coherente : 300 s sur 1800 s -> ~1/6 des battements en adaptation
ratio = len(s.yad)/(len(s.yad)+len(s.yte))
assert 0.13 < ratio < 0.20, ratio
# 5. formes : contexte a 3 canaux
assert s.Xtr.shape[1:] == (C.RESAMPLE_LEN, 3) and s.Rtr.shape[1] == C.RR_DIM
# 6. scaler ajuste sur le train : train centre-reduit, test NON force a 0
assert abs(s.Rtr.mean()) < 0.05
# 7. sans adaptation, tout DS2 est du test
s2 = build_split(X, RR, y, g, t, dur, use_adaptation=False, verbose=False)
assert len(s2.yad) == 0 and len(s2.yte) == int(np.isin(g, C.DS2).sum())
# 8. duree evaluee amputee des 5 minutes d'adaptation
d = test_durations_after_adaptation(dur, True)
assert all(abs(v - 1500.0) < 1e-9 for v in d.values())
print("ratio adaptation = %.3f | duree evaluee/enr. = %.0f s" % (ratio, list(d.values())[0]))
print("\nLOGIQUE TRAIN / VALIDATION / TEST / ADAPTATION : TOUS LES TESTS PASSENT")
