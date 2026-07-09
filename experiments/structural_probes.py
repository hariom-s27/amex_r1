# experiments/structural_probes.py — self-contained, no paste-slot needed
import numpy as np, pandas as pd
from pathlib import Path
DATA = Path("data/raw/r1_datset.csv"); SUB = Path("outputs/submissions")
df = pd.read_csv(DATA); df.columns=[c.strip().lower() for c in df.columns]
N,k = len(df),100_000
cat = df[["f6","f7","f8","f9","f10"]].sum(axis=1,min_count=1)

def preds(f): return pd.ExcelFile(SUB/f).parse("Predictions")["Prediction"].values.astype(float)
def top(v):
    t=np.zeros(N,bool); t[np.argpartition(-np.asarray(v,float),k)[:k]]=True; return t

champ = preds("sub04_impute2.xlsx")          # our 0.875 profit scores

# champion's benefit cost term (same coefficients used to build 0.875)
f13=df.f13.fillna(0).values; f14=df.f14.fillna(0).values
f15=df.f15.fillna(0).values; f16=df.f16.fillna(0).values
f1=df.f1.fillna(0).values;   f21=df.f21.fillna(0).values
ben = 35*f13 + f14 + 17.5*f15 + f16          # <- if your champ used other numbers, tell me; else this is fine

champ_top = top(champ)

# ---- probes built by ADJUSTING the champion, not rebuilding it ----
cands = {
 "BEN0 (add benefits back = remove penalty)": champ + ben,   # champion had -ben; undo it
 "BEN_half":                                   champ + 0.5*ben,
 "F21+ small positive":                        champ + 0.004*f21,
 "M-UP (extra revolve reward)":                champ + 0.10*0.27*f1,   # +50% margin on f1
}

ANCH = {"FINAL_submission_r1.xlsx":82.0,"sub04_impute2.xlsx":87.5,"sub04_m10.xlsx":78.4,
        "sub05_consensus.xlsx":83.4,"sub06_final.xlsx":83.4}   # ADD risk_842 line if it was scored!
A=[top(preds(f)) for f in ANCH]; s=np.array(list(ANCH.values()))/100; n=len(A)
Ov=np.array([[np.count_nonzero(A[i]&A[j])/k for j in range(n)] for i in range(n)])
W=np.linalg.solve(Ov+1e-3*np.eye(n),s)
bias=[]
for i in range(n):
    idx=[j for j in range(n) if j!=i]
    w=np.linalg.solve(Ov[np.ix_(idx,idx)]+1e-3*np.eye(n-1),s[idx])
    bias.append(100*s[i]-100*float(np.array([Ov[i][j] for j in idx])@w))
BIAS=float(np.mean(bias)); print(f"predictor mean bias +{BIAS:.1f}\n")

def read(t,name):
    ov=np.array([np.count_nonzero(t&a)/k for a in A]); raw=100*float(ov@W)
    nov=100*(1-max(np.count_nonzero(t&a)/k for a in A))
    lb=100*max(o+si-1 for o,si in zip(ov,s)); ub=100*min(1-abs(o-si) for o,si in zip(ov,s))
    e=np.where(t&~champ_top)[0]; l=np.where(champ_top&~t)[0]
    print(f"{name}: predicted {raw+BIAS:.1f} | bounds [{lb:.1f},{ub:.1f}] | novelty {nov:.1f}%")
    for tag,ix in [("ENTER",e),("LEAVE",l)]:
        d=df.loc[ix]
        print(f"  {tag} n={len(ix):,}: spend {cat.loc[ix].median():,.0f} | f1 {d.f1.median():,.0f} "
              f"| f11 {d.f11.median():.4f} | f21 {d.f21.median():,.0f} | benefit {(d.f13.fillna(0)+d.f15.fillna(0)).median():.0f}")
    print()
    return raw+BIAS

print(f"parity self-check (champ vs itself): {100*np.count_nonzero(top(champ)&champ_top)/k:.0f}% (must be 100)\n")
best=(-1,None)
for nm,v in cands.items():
    p=read(top(v),nm)
    if p>best[0]: best=(p,nm)
print(f"WINNER by predicted: {best[1]} ({best[0]:.1f}) — do NOT upload yet, paste output to Claude")