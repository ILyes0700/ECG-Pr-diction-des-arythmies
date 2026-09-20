import sys, numpy as np
sys.path.insert(0, ".")
from src.calibration import softmax, fit_temperature, apply_temperature, negative_log_likelihood
from src.uncertainty import predictive_entropy, deferral_for_target_accuracy
sys.path.insert(0,"src"); import clinical_metrics as cmx

rng = np.random.default_rng(0)
n, k = 4000, 5
y = rng.integers(0, k, n)
# logits sur-confiants (typique d'un reseau non calibre) -> T optimal > 1
base = rng.normal(size=(n, k))
base[np.arange(n), y] += 2.0
logits = base * 3.0
T, nll = fit_temperature(logits, y)
assert T > 1.2, T
p_raw, p_cal = softmax(logits), apply_temperature(logits, T)
e_raw, _ = cmx.calibration_errors(y, p_raw)
e_cal, _ = cmx.calibration_errors(y, p_cal)
assert e_cal["ECE"] < e_raw["ECE"]
assert negative_log_likelihood(p_cal, y) <= negative_log_likelihood(p_raw, y)
# la calibration ne change PAS les predictions (donc ni Se ni Sp)
assert (p_raw.argmax(1) == p_cal.argmax(1)).all()
print("temperature scaling OK  T=%.2f  ECE %.4f -> %.4f (predictions inchangees)" % (T, e_raw["ECE"], e_cal["ECE"]))

ent = predictive_entropy(p_cal)
assert ent.min() >= 0 and ent.max() <= np.log(k) + 1e-6
d = deferral_for_target_accuracy(y, p_cal, ent, 0.99)
assert 0 <= d["taux_renvoi"] <= 1
print("option de rejet OK  couverture=%.2f pour 99%% de fiabilite" % d["couverture"])
print("\nTESTS CALIBRATION / INCERTITUDE : OK")
