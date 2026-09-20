import sys, numpy as np
sys.path.insert(0,".")
from src.quality import beat_quality, quality_mask, quality_report, clean_training_set
from src import config as C

rng = np.random.default_rng(0); L = 128
def qrs(n):                                   # battement propre : pic net + ligne de base
    t = np.linspace(-1,1,L); base = 0.05*np.sin(2*np.pi*t)
    x = np.stack([base + 1.6*np.exp(-((t-0.0)**2)/0.0009) for _ in range(n)])
    return (x + rng.normal(0,0.01,(n,L)))
propres = qrs(300)
plats = np.zeros((20,L))
bruit = rng.normal(0,1,(20,L))                # bruit gaussien pur -> kurtosis ~ 3 mais HF eleve
nans = np.full((5,L), np.nan)
satures = np.clip(qrs(15)*8, -2, 2)

X = np.concatenate([propres, plats, bruit, nans, satures])
keep, q = quality_mask(X)
print("propres gardes : %d/300" % keep[:300].sum())
print("plats gardes   : %d/20"  % keep[300:320].sum())
print("bruit garde    : %d/20"  % keep[320:340].sum())
print("NaN gardes     : %d/5"   % keep[340:345].sum())
print("satures gardes : %d/15"  % keep[345:].sum())
assert keep[:300].mean() > 0.95      # on ne jette pas les bons battements
assert keep[300:320].sum() == 0      # plats rejetes
assert keep[340:345].sum() == 0      # NaN rejetes
assert keep[320:340].mean() < 0.3    # bruit majoritairement rejete

# rapport par classe
y = np.zeros(len(X), int); y[300:320]=1; y[320:340]=2; y[340:345]=3; y[345:]=4
rep, k2 = quality_report(X, y, C.CLASS_NAMES, "train")
print("\n", rep.to_string(index=False))

# nettoyage du train : protege les classes rares
Xr = np.concatenate([propres, plats[:3]]); yr = np.array([0]*300 + [3]*3)
Rr = rng.normal(size=(len(Xr),6)); gr = np.array(["100"]*len(Xr))
Xc, Rc, yc, gc, kc = clean_training_set(Xr, Rr, yr, gr)
assert (yc==3).sum() == 3, "la classe rare F a ete supprimee : interdit"
print("\nprotection des classes rares OK — F conservee malgre la mauvaise qualite")
print("\nTESTS QUALITE : OK")
