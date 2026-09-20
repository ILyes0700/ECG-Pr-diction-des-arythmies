import sys, numpy as np
sys.path.insert(0,"src")
import preprocessing as pp

# z-score par battement : moyenne 0, ecart-type 1, invariant au gain
b = np.array([1.,2.,3.,4.,10.])
z = pp.zscore_beat(b); z2 = pp.zscore_beat(b*7.5+3)
assert abs(z.mean())<1e-6 and abs(z.std()-1)<1e-6 and np.allclose(z,z2,atol=1e-5)
print("zscore_beat OK (invariant au gain et a l'offset)")

# RR : battement premature detecte par pre/local < 1
fs=360
peaks = np.cumsum([0]+[360]*10+[200]+[360]*10)     # un battement premature au 12e
R = pp.rr_features(peaks, fs)
assert R.shape==(len(peaks),6)
i = 11
assert R[i,3] < 0.75, R[i,3]        # ratio pre/local effondre = prematurite
assert R[i-1,3] > 0.95
print("rr_features OK  ratio pre/local du battement premature = %.3f"%R[i,3])

# contexte : pas de franchissement de patient
beats = np.arange(12, dtype=np.float32).reshape(6,2)
groups = np.array(["A","A","A","B","B","B"])
C = pp.stack_context_by_record(beats, groups)
assert C.shape==(6,2,3)
assert np.allclose(C[0,:,0], beats[0])              # bord replique
assert np.allclose(C[3,:,0], beats[3])              # debut de B ne voit pas la fin de A
assert np.allclose(C[2,:,2], beats[2])              # fin de A ne voit pas le debut de B
print("stack_context_by_record OK (aucune fuite entre patients)")

# augmentation : forme conservee, train seulement, deterministe avec seed
rng = np.random.default_rng(0)
X = np.random.RandomState(0).randn(20,128,3).astype(np.float32)
A1 = pp.augment_batch(X, np.random.default_rng(0))
A2 = pp.augment_batch(X, np.random.default_rng(0))
assert A1.shape==X.shape and np.allclose(A1,A2) and not np.allclose(A1,X)
print("augment_batch OK (reproductible, formes conservees)")

# RRScaler : ajuste sur le train, applique au test sans refit
s = pp.RRScaler().fit(np.random.RandomState(1).randn(100,6)*3+5)
t = s.transform(np.random.RandomState(2).randn(50,6)*3+5)
assert abs(t.mean())<0.6 and t.shape==(50,6)
assert not hasattr(s,"fitted_on_test")
print("RRScaler OK")
print("\nTOUS LES TESTS DE PRETRAITEMENT PASSENT")
