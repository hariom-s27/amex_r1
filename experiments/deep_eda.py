"""experiments/deep_eda.py — understand the data STRUCTURE before more models (research-guided)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, build, top_set, overlap

df = load_data(); arr = prep(df); spend = imputed_spend(df)
R = lambda x: pd.Series(np.asarray(x, float)).rank(pct=True).values
champ_v = pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['Prediction'].values
champ = top_set(champ_v)

# feature matrix on RANKS (scale-free, robust) — the columns that carry economics
cols = ["f1","f11","f13","f14","f15","f16","f4","f12","f19","f22"]
X = np.column_stack([R(df[c].fillna(0).values) for c in cols] + [R(spend)])
Xcols = cols + ["spend"]

print("=== STUDY 1 — how many natural customer segments? ===")
for kk in [3,4,5,6]:
    km = KMeans(n_clusters=kk, n_init=5, random_state=0).fit(X)
    inertia = km.inertia_
    print(f"  k={kk}: inertia={inertia:,.0f}")
print("  (look for the 'elbow' where inertia stops dropping fast)")

K_SEG = 5
km = KMeans(n_clusters=K_SEG, n_init=10, random_state=0).fit(X)
seg = km.labels_

print(f"\n=== STUDY 2 — champion's top-20% by segment (k={K_SEG}) ===")
in_top = np.zeros(C.N, bool); in_top[list(champ)] = True
for s in range(K_SEG):
    m = seg==s
    print(f"  seg{s}: n={m.sum():>7,} ({100*m.mean():4.1f}%) | "
          f"in champ-top {100*in_top[m].mean():4.1f}% | "
          f"spend {df.catsum[m].median():>8,.0f} | f1 {df.f1[m].median():>7,.0f} | "
          f"f11 {df.f11[m].median():.4f} | benefit {(df.f13+df.f15)[m].median():.0f}")
print("  -> a segment with HIGH spend but LOW champ-top% = members we may be wrongly excluding")

print("\n=== STUDY 3 — which variables DEFINE the structure? (drop-one clustering) ===")
base_km = KMeans(n_clusters=K_SEG, n_init=5, random_state=0).fit(X)
from sklearn.metrics import adjusted_rand_score
for i,c in enumerate(Xcols):
    Xi = np.delete(X, i, axis=1)
    kmi = KMeans(n_clusters=K_SEG, n_init=5, random_state=0).fit(Xi)
    ari = adjusted_rand_score(base_km.labels_, kmi.labels_)
    print(f"  drop {c:6}: structure similarity {ari:.3f}  (LOW = {c} strongly shapes segments)")

print("\n=== STUDY 4 — who lives at the champion's boundary? ===")
cut = np.quantile(champ_v, 0.8)
span = np.quantile(champ_v,0.99)-np.quantile(champ_v,0.01)
near = np.abs(champ_v-cut) <= 0.01*span
print(f"  boundary members: {near.sum():,}")
g = df[near]
print(f"  their profile: spend {g.catsum.median():,.0f} | f1 {g.f1.median():,.0f} | "
      f"f11 {g.f11.median():.4f} | revolver {100*(g.f1>0).mean():.0f}%")

print("\n=== STUDY 5 — segment overlap with Class-B (our diverse model) ===")
classB = R(spend)+R(df.f1.fillna(0))-R(df.f11.fillna(0))+0.5*R(df.f13.fillna(0)+df.f15.fillna(0))
tB = top_set(classB); inB=np.zeros(C.N,bool); inB[list(tB)]=True
for s in range(K_SEG):
    m=seg==s
    print(f"  seg{s}: champ-top {100*in_top[m].mean():4.1f}% vs classB-top {100*inB[m].mean():4.1f}%  "
          f"(big gap = the two models disagree about this segment)")