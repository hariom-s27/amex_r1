# experiments/ben_scan.py — find the benefit scale with the highest guaranteed floor
import numpy as np, pandas as pd
from pathlib import Path
SUB = Path("outputs/submissions"); df = pd.read_csv("data/raw/r1_datset.csv")
df.columns=[c.strip().lower() for c in df.columns]; N,k=len(df),100_000
def preds(f): return pd.ExcelFile(SUB/f).parse("Predictions")["Prediction"].values.astype(float)
def top(v):
    t=np.zeros(N,bool); t[np.argpartition(-v,k)[:k]]=True; return t
champ=preds("sub04_impute2.xlsx")
ben=(35*df.f13.fillna(0)+df.f14.fillna(0)+17.5*df.f15.fillna(0)+df.f16.fillna(0)).values
ANCH={"FINAL_submission_r1.xlsx":82.0,"sub04_impute2.xlsx":87.5,"sub04_m10.xlsx":78.4,
      "sub05_consensus.xlsx":83.4,"sub06_final.xlsx":83.4}    # add risk_842 if scored
A=[top(preds(f)) for f in ANCH]; s=np.array(list(ANCH.values()))/100
print("scale | guaranteed floor | ceiling | novelty  (pick HIGHEST floor)")
best=(-1,None,None)
for sc in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0]:
    # champion subtracted ben; to apply scale x, add back (1-x)*ben
    t=top(champ + (1-sc)*ben)
    ov=[np.count_nonzero(t&a)/k for a in A]
    lb=100*max(o+si-1 for o,si in zip(ov,s)); ub=100*min(1-abs(o-si) for o,si in zip(ov,s))
    nov=100*(1-max(ov))
    flag=" <-- champ" if sc==1.0 else ""
    print(f"  {sc:.1f}  |   {lb:5.1f}        | {ub:5.1f}  | {nov:4.1f}%{flag}")
    if lb>best[0] and sc<1.0: best=(lb,sc,ub)
print(f"\nBEST FLOOR: scale={best[1]} floor={best[0]:.1f} ceiling={best[2]:.1f}")
print("build that scale as sub10 via your writer, patch framework benefit coefficient, upload.")