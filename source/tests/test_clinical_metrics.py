import sys, numpy as np
sys.path.insert(0, "src")
import clinical_metrics as cmx

CN = np.array(["N","S","V","F","Q"])
rng = np.random.default_rng(0)

# --- 1. matrice de confusion et comptes binaires -----------------------------
y = np.array([0,0,1,1,2,2,2,3,4])
p = np.array([0,1,1,0,2,2,0,3,4])
cm = cmx.confusion_matrix(y,p,5)
assert cm.sum()==9 and cm[0,0]==1 and cm[0,1]==1
tp,fp,fn,tn = cmx.binary_counts(cm,2)
assert (tp,fn)==(2,1), (tp,fp,fn,tn)
assert tp+fp+fn+tn==9

# --- 2. metriques cliniques : cas connu --------------------------------------
# 100 positifs classe V dont 90 detectes ; 900 negatifs dont 45 faux positifs
y2 = np.array([2]*100 + [0]*900)
p2 = np.array([2]*90 + [0]*10 + [2]*45 + [0]*855)
t = cmx.clinical_table(y2,p2,CN)
r = t[t.Classe=="V"].iloc[0]
assert abs(r.Se-0.90)<1e-9 and abs(r.Sp-0.95)<1e-9
assert abs(r.VPP-90/135)<1e-9
assert abs(r["LR+"]-18.0)<1e-6 and abs(r["LR-"]-(0.10/0.95))<1e-9
assert abs(r["Youden_J"]-0.85)<1e-9 and abs(r["NND"]-1/0.85)<1e-9
# VPP ajuste a 5% de prevalence
expected = 0.9*0.05/(0.9*0.05+0.05*0.95)
assert abs(r["VPP@prev5%"]-expected)<1e-9
print("clinical_table OK  Se=%.3f Sp=%.3f LR+=%.2f DOR=%.1f MCC=%.3f"%(r.Se,r.Sp,r["LR+"],r.DOR,r.MCC))

# --- 3. EC57 gross vs average -------------------------------------------------
g = np.array(["A"]*500+["B"]*500)
ec = cmx.ec57_statistics(y2,p2,g,CN)
assert set(ec.Classe)=={"S","V"}
print("EC57 OK\n", ec.round(4).to_string(index=False))

# --- 4. alarmes par heure -----------------------------------------------------
dur = {"A":1800.0,"B":1800.0}
ab = cmx.alarm_burden(y2,p2,g,dur,CN)
glob = ab[ab.Enregistrement=="GLOBAL"].iloc[0]
assert abs(glob.Heures-1.0)<1e-9 and abs(glob.FA_par_heure-45.0)<1e-9
print("alarm_burden OK  FA/h=%.1f  manques/h=%.1f"%(glob.FA_par_heure, glob.Manques_par_heure))

# --- 5. episodes --------------------------------------------------------------
yt = np.zeros(50,int); yt[10:15]=2; yt[30:33]=2
yp = np.zeros(50,int); yp[12:15]=2                      # 1er episode detecte a t+2
ep = cmx.episode_analysis(yt,yp,np.array(["A"]*50),CN)
assert ep["Episodes_reels"]==2 and ep["Episodes_detectes"]==1
assert ep["Latence_mediane_battements"]==2.0
print("episode_analysis OK", ep)

# --- 6. calibration -----------------------------------------------------------
proba = np.zeros((1000,5)); proba[np.arange(1000), y2]=1.0     # parfait et sur-confiant
errs,_ = cmx.calibration_errors(y2,proba)
assert errs["ECE"]<1e-9
proba2 = np.full((1000,5),0.2)
b = cmx.brier_score(y2,proba2,5)
assert abs(b-(0.8**2+4*0.2**2))<1e-9
print("calibration OK  ECE_parfait=%.2e  Brier_uniforme=%.3f"%(errs["ECE"],b))

# --- 7. decision curve --------------------------------------------------------
pv = np.where(y2==2, 0.9, 0.1)
dc = cmx.decision_curve((y2==2).astype(int), pv)
assert dc.NB_modele.max() > dc.NB_traiter_tous.max()
print("decision_curve OK  NB_max_modele=%.4f"%dc.NB_modele.max())

# --- 8. seuil a sensibilite cible --------------------------------------------
probs = np.concatenate([rng.uniform(0.4,1.0,100), rng.uniform(0.0,0.5,900)])
yb = np.array([1]*100+[0]*900)
th = cmx.threshold_for_target_sensitivity(yb,probs,0.95)
se = ((probs>=th)&(yb==1)).sum()/100
assert se>=0.94, se
print("seuil Se>=95%% OK  th=%.3f  Se_obtenue=%.2f"%(th,se))

# --- 9. couverture / risque ---------------------------------------------------
crc = cmx.coverage_risk_curve(y2, np.clip(proba-0.05,0.01,1))
assert crc.Couverture.is_monotonic_increasing
print("coverage_risk OK")

# --- 10. cout clinique --------------------------------------------------------
C = np.zeros((4,4)); C[2,0]=10.0
cc = cmx.clinical_cost(y2,p2,C)
assert abs(cc["Cout_clinique_total"]-100.0)<1e-9   # 10 V rates * 10
print("clinical_cost OK", cc)

# --- 11. bootstrap + McNemar --------------------------------------------------
bs = cmx.bootstrap_metric(y2,p2,lambda a,b: cmx.macro_f1(a,b,5), n=200, seed=1)
assert bs["ci95_low"] <= bs["mean"] <= bs["ci95_high"]
mn = cmx.mcnemar_test(y2,p2,np.zeros_like(p2))
assert mn["p_value"] < 0.05
print("bootstrap OK", {k:round(v,4) for k,v in bs.items()}, "| McNemar p=%.2e"%mn["p_value"])

print("\nTOUS LES TESTS DE METRIQUES CLINIQUES PASSENT")
