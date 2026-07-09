"""experiments/batch_engagement_scaled.py — CORRECTLY scaled engagement test.
Prior f12/f22 probes had 77% novelty = scale bug (term too big). Use TINY weights."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, sink_spend
from src.scoring import top_set, overlap
import src.submit as sb

df = load_data(); sp = sink_spend(df); ids = df["id"].values
catmiss = df[C.CATS].isna().all(axis=1).values
f1=df.f1.fillna(0).values; f11=df.f11.fillna(0).values
f13=df.f13.fillna(0).values; f14=df.f14.fillna(0).values
f15=df.f15.fillna(0).values; f16=df.f16.fillna(0).values
f12=df.f12.fillna(0).values; f22=df.f22.fillna(0).values; f2=df.f2.fillna(0).values

BEN = 35*f13 + f14 + 40*f15 + f16
base_v = 0.025*sp + 0.35*f1 - 1.0*BEN - 1.5*f11*(f1+sp/12) - 1e9*catmiss
base = base_v.copy(); bt = top_set(base)

R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
S = np.std(base_v[~catmiss])   # scale reference from NON-block members only
# TINY weights so engagement REFINES, not dominates (target novelty 2-10%)
probes = {
    "f12_tiny_+":  base_v + 0.05*S*R(f12),   # small login reward
    "f12_tiny_-":  base_v - 0.05*S*R(f12),   # small login penalty
    "f22_tiny_-":  base_v - 0.05*S*R(f22),
    "f2_tiny_-":   base_v - 0.05*S*R(f2),    # small cancel-call penalty
    "logins_x_spend": base_v + 0.00000003*sp*R(f12),  # engaged spenders
}
print("SCALED ENGAGEMENT PROBES (target 2-10% novelty = refinement, not takeover):")
built=[]
for name,v in probes.items():
    t=top_set(v); nov=100-overlap(t,bt)
    ge=df.iloc[list(t-bt)] if len(t-bt)>0 else df.iloc[:1]
    flag = "  <-- GOOD refinement" if 2<=nov<=10 else ("  <-- inert" if nov<2 else "  <-- still too big")
    print(f"  {name:16} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f}{flag}")
    if 1.0 <= nov <= 12:
        sb.write_sub(v, ids, C.OUT/f"sub20_{name}.xlsx", f"s20_{name}")
        sb.framework_guard(C.OUT/f"sub20_{name}.xlsx")
        built.append(name)
print(f"\nbuilt: {built}")
print("upload the one with sensible persona (ENTER spend should stay HIGH ~80-95K)")