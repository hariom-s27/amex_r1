"""experiments/build_risk_candidates.py — build moderate & strong risk files + compare profiles."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, top_set, overlap
from src.submit import write_sub, framework_guard

df = load_data(); arr = prep(df); spend = imputed_spend(df)
R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
ids = df["id"].values
f1=df.f1.fillna(0).values; f11=df.f11.fillna(0).values
champ_v = pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['Prediction'].values
champ = top_set(champ_v)

def risk_model(lgd):
    return 0.025*spend + 0.22*f1 - lgd*f11*(f1 + spend/12)

mod = risk_model(2.5)      # moderate
strong = risk_model(6.0)   # strong

# ---- rebuild the 5 segments (same as deep_eda) for profiling ----
cols = ["f1","f11","f13","f14","f15","f16","f4","f12","f19","f22"]
X = np.column_stack([R(df[c].fillna(0).values) for c in cols] + [R(spend)])
seg = KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(X)

def profile(name, score):
    t = top_set(score); nov = 100 - overlap(t, champ)
    inT = np.zeros(C.N,bool); inT[list(t)]=True
    print(f"\n{name}  (novelty vs champion {nov:.1f}%)")
    for s in range(5):
        m = seg==s
        print(f"  seg{s}: picked {100*inT[m].mean():4.1f}% | spend {df.catsum[m].median():>7,.0f} "
              f"| f1 {df.f1[m].median():>6,.0f} | f11 {df.f11[m].median():.4f}")

inC=np.zeros(C.N,bool); inC[list(champ)]=True
print("SEGMENT PICK-RATES (what fraction of each segment enters the top-20%)")
print("\nCHAMPION 0.875 (reference)")
for s in range(5):
    m=seg==s
    print(f"  seg{s}: picked {100*inC[m].mean():4.1f}% | spend {df.catsum[m].median():>7,.0f} "
          f"| f1 {df.f1[m].median():>6,.0f} | f11 {df.f11[m].median():.4f}")
profile("MODERATE risk (lgd=2.5)", mod)
profile("STRONG risk (lgd=6.0)", strong)

# ---- who moves? enter/leave personas vs champion ----
def moves(name, score):
    t = top_set(score)
    enter=list(t-champ); leave=list(champ-t)
    ge=df.iloc[enter]; gl=df.iloc[leave]
    print(f"\n{name} MOVES:")
    print(f"  ENTER n={len(enter):,}: spend {ge.catsum.median():>7,.0f} | f1 {ge.f1.median():>6,.0f} | f11 {ge.f11.median():.4f} | revolver {100*(ge.f1>0).mean():.0f}%")
    print(f"  LEAVE n={len(leave):,}: spend {gl.catsum.median():>7,.0f} | f1 {gl.f1.median():>6,.0f} | f11 {gl.f11.median():.4f} | revolver {100*(gl.f1>0).mean():.0f}%")
moves("MODERATE", mod)
moves("STRONG", strong)

# ---- build BOTH submission files (framework-correct, validated) ----
write_sub(mod,    ids, C.OUT/"sub07_risk_moderate.xlsx", "s07_risk_lgd25")
framework_guard(C.OUT/"sub07_risk_moderate.xlsx")
write_sub(strong, ids, C.OUT/"sub08_risk_strong.xlsx",   "s08_risk_lgd60")
framework_guard(C.OUT/"sub08_risk_strong.xlsx")
print("\nBoth files built, framework-checked. Neither uploaded yet — you choose after seeing profiles.")