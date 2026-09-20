import sys, numpy as np
sys.path.insert(0, ".")
from src import config as C
from src import evaluate as E

rng = np.random.default_rng(0)
n = 5000
y = rng.choice(C.N_CLASSES, n, p=[.86,.03,.08,.03])
# probabilites synthetiques correlees a la verite
proba = rng.dirichlet(np.ones(C.N_CLASSES)*0.6, n)
proba[np.arange(n), y] += 1.6
proba /= proba.sum(1, keepdims=True)

# ROC/PR/AUC
yb = (y==2).astype(int)
fpr,tpr = E.roc_curve_np(yb, proba[:,2]); a = E.auc_np(fpr,tpr)
assert 0.9 < a <= 1.0, a
# AUC d'un score aleatoire ~ 0.5
fpr2,tpr2 = E.roc_curve_np(yb, rng.random(n))
assert 0.45 < E.auc_np(fpr2,tpr2) < 0.55
print("ROC/AUC OK  AUC_signal=%.3f  AUC_aleatoire=%.3f"%(a, E.auc_np(fpr2,tpr2)))

g, ct = E.global_metrics(y, proba)
assert 0 < g["F1_macro"] <= 1 and 0 < g["Accuracy"] <= 1
assert set(["Se_S","PP_S","Se_V","PP_V","Brier","ECE","MCE","Cout_clinique_total"]) <= set(g)
print("global_metrics OK", {k:round(v,3) for k,v in g.items() if k in ("Accuracy","F1_macro","Se_V","Se_S","Brier","ECE")})

# kappa : parfait = 1, aleatoire ~ 0
assert abs(E.cohen_kappa(y,y)-1) < 1e-9
assert abs(E.cohen_kappa(y, rng.permutation(y))) < 0.05
print("kappa OK")

# point de fonctionnement clinique : la sensibilite V doit augmenter
half = n//2
tab, th, pred_clin = E.table_operating_point(y[:half], proba[:half], y[half:], proba[half:])
se_argmax = tab.iloc[0].Se_V; se_clin = tab.iloc[1].Se_V
assert se_clin >= se_argmax
print(tab.round(4).to_string(index=False))
print("seuils cliniques (choisis sur validation) : V=%.3f S=%.3f"%(th[C.IDX_V], th[C.IDX_S]))

# significativite
worse = proba.copy(); worse[:,2] *= 0.3; worse /= worse.sum(1,keepdims=True)
sig, ci = E.table_significance(y, proba, {"baseline_degradee": worse}, n_boot=100)
assert sig.iloc[0]["Significatif_5%"]=="oui"
print(sig.to_string(index=False))

# tables dataset / patients
g_ids = np.array(C.DS1*(n//len(C.DS1)+1))[:n]
t1 = E.table_dataset(y, g_ids); assert t1.iloc[-1].Total == n
pp = E.table_per_patient(y, proba.argmax(1), g_ids); assert len(pp)==len(set(g_ids))
print("tables dataset/patients OK")
print("\nTESTS EVALUATION : OK")
