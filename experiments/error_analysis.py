# experiments/error_analysis.py  — learn from our BEST model, build nothing new
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "submissions"
df = pd.read_csv(ROOT / "data" / "raw" / "r1_datset.csv"); df.columns=[c.strip().lower() for c in df.columns]
N,k = len(df),100_000
cat = df[["f6","f7","f8","f9","f10"]].sum(axis=1,min_count=1)

def top(p): 
    v=pd.ExcelFile(p).parse("Predictions")["Prediction"].values
    t=np.zeros(N,bool); t[np.argpartition(-v,k)[:k]]=True; return t,v
best,_ = top(OUT/"sub04_impute2.xlsx")     # 0.875
anch = {"v1":(OUT/"FINAL_submission_r1.xlsx",82.0),
        "m10":(OUT/"sub04_m10.xlsx",78.4),
        "cons":(OUT/"sub05_consensus.xlsx",83.4)}

# 1. Stable vs fragile seats: does best agree with the other models?
agree = np.zeros(N,int)
for p,_ in anch.values():
    t,_=top(p); agree += t.astype(int)
print("Of best's 100k top seats, how many OTHER models agree:")
for a in range(4):
    m = best & (agree==a)
    print(f"  {a} others agree: {m.sum():,} seats | med catsum {cat[m].median():,.0f} | med f1 {df.f1[m].median():,.0f} | rev% {(df.f1[m]>0).mean()*100:.0f}")

# 2. The boundary zone — where errors concentrate
v=pd.ExcelFile(OUT/"sub04_impute2.xlsx").parse("Predictions")["Prediction"].values
cut=np.sort(v)[::-1][k-1]; band=np.abs(v-cut)<=0.01*(np.nanpercentile(v,99)-np.nanpercentile(v,1))
print(f"\nboundary band: {band.sum():,} members fighting for the cutoff")
print(f"  their profile: med catsum {cat[band].median():,.0f} | med f1 {df.f1[band].median():,.0f} | "
      f"cats-missing {df.f6[band].isna().mean()*100:.0f}% | enrolled {df.f4[band].notna().mean()*100:.0f}%")

# 3. Imputed members near the boundary = our biggest uncertainty
imp_band = band & df.f6.isna().values
print(f"\n  imputed members in the boundary band: {imp_band.sum():,} "
      f"(these are our RISKIEST guesses — we don't know their real spend)")

# 4. Feature spread of top vs bottom (what actually drives our ranking?)
print("\nwhat separates our top from our cutoff neighbours:")
for c in ["f5","f1","f11","f21","f13"]:
    hi=df[c][best].median(); lo=df[c][band].median()
    print(f"  {c}: top {hi:,.1f} vs boundary {lo:,.1f}")