"""experiments/better_imputer.py — research-backed stronger imputation of the 23% block."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, top_set, overlap
from src.submit import write_sub, framework_guard

df = load_data(); arr = prep(df); ids = df["id"].values
R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
champ_v = pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['Prediction'].values
champ = top_set(champ_v)

# richer feature set for imputing catsum (research: more predictors = better)
feats = ["f1","f4","f21","f13","f14","f15","f16","f11","f12","f22","f19","f20","f2","f3"]
okc = df["catsum"].notna()
X = df[feats].rank(); X = X.fillna(X.median())
y = df.loc[okc,"catsum"].rank()
idx = np.random.default_rng(0).permutation(okc.sum()); cut=int(0.8*len(idx))
Xtr,Xte = X[okc.values].iloc[idx[:cut]], X[okc.values].iloc[idx[cut:]]
ytr,yte = y.iloc[idx[:cut]], y.iloc[idx[cut:]]
def rho(a,b): return pd.Series(a).rank().corr(pd.Series(b).rank())

print("IMPUTER COMPARISON (holdout rank-corr — higher = better):")
models = {
    "gbm (current)":   HistGradientBoostingRegressor(max_iter=300,random_state=0),
    "gbm deep":        HistGradientBoostingRegressor(max_iter=600,max_depth=8,learning_rate=0.05,random_state=0),
    "extra-trees":     ExtraTreesRegressor(n_estimators=200,n_jobs=-1,random_state=0),
    "knn-15":          KNeighborsRegressor(n_neighbors=15),
}
best=None
for n,m in models.items():
    m.fit(Xtr,ytr); r=rho(m.predict(Xte),yte)
    print(f"  {n:16} {r:.3f}")
    if best is None or r>best[0]: best=(r,n,m)
print(f"  BEST: {best[1]} ({best[0]:.3f}) vs current gbm ~0.64")

# build champion with the BEST imputer
m=best[2]; pct=pd.Series(m.predict(X),index=df.index).rank(pct=True).values
srt=np.sort(df["catsum"].dropna().values)
imp=srt[np.clip((pct*len(srt)).astype(int),0,len(srt)-1)]
spend_new=np.where(df["catsum"].notna(), df["catsum"].clip(lower=0).fillna(0), imp)
f1=df.f1.fillna(0).values; f11=df.f11.fillna(0).values
new = 0.025*spend_new + 0.22*f1 - 0.85*f11*(f1+spend_new/12)   # champion shape, better imputer
t=top_set(new); nov=100-overlap(t,champ)
print(f"\nnew-imputer champion: novelty vs current champion {nov:.1f}%")
print("  (novelty 5-15% = meaningfully different imputation; worth the slot if imputer improved)")

if best[0] > 0.68:   # only build if the imputer is genuinely better
    write_sub(new, ids, C.OUT/"sub10_better_imputer.xlsx","s10_better_imputer")
    framework_guard(C.OUT/"sub10_better_imputer.xlsx")
    print("  BUILT sub10 — imputer improved enough to justify a slot.")
else:
    print("  NOT built — no imputer beat current meaningfully. Bank the ensemble instead.")