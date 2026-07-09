"""experiments/batch_explore.py — build 5 single-lever probes on the sub13b champion basis.
Each isolates ONE untested lever. Upload the best-audited ones; each result teaches one thing."""
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
f2=df.f2.fillna(0).values; f3=df.f3.fillna(0).values
BEN=(35*df.f13.fillna(0)+df.f14.fillna(0)+17.5*df.f15.fillna(0)+df.f16.fillna(0)).values
t23=df.f23.fillna(df.f23.median()).values
veteran = (t23==np.nanmax(df.f23.values))   # 4,447 longest-tenure

# CHAMPION BASIS = sub13b: block-excluded, m=0.22, lgd=0.85, ben=-1.0
def champ_basis():
    v = 0.025*sp + 0.22*f1 - 1.0*BEN - 0.85*f11*(f1+sp/12)
    return v - 1e9*catmiss
base = champ_basis(); bt = top_set(base)
champ_v = pd.ExcelFile(C.OUT/"sub13b_evict_only.xlsx").parse("Predictions")["Prediction"].values
assert overlap(bt, top_set(champ_v)) > 99, "basis must match sub13b"

R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
probes = {}
# LEVER A: m-ratio UP — but MEASURED (small step), not fence. Tests right side of peak.
probes["m028"] = 0.025*sp + 0.28*f1 - 1.0*BEN - 0.85*f11*(f1+sp/12) - 1e9*catmiss
# LEVER B: collections/calls removal (never board-tested cleanly)
probes["no_collcall"] = base + 0.0  # base already has no coll/call — skip if identical
probes["with_collcall"] = 0.025*sp + 0.22*f1 - 1.0*BEN - 0.85*f11*(f1+sp/12) - 500*f3 - 75*f2 - 1e9*catmiss
# LEVER C: veteran/tenure boost (welcome-bonus recovered for long-tenure)
vb = base.copy(); vb[veteran] += 0.5*np.std(base[~catmiss]); probes["veteran_boost"] = vb
# LEVER D: heavier revolve via rewards multiplier on spend (5x travel per product sheet)
probes["spend_light"] = 0.020*sp + 0.22*f1 - 1.0*BEN - 0.85*f11*(f1+sp/12) - 1e9*catmiss

print("PROBE PROFILES (novelty vs sub13b champion):")
for name,v in probes.items():
    t=top_set(v); nov=100-overlap(t,bt)
    if nov < 0.3:
        print(f"  {name:16} novelty {nov:.1f}%  <-- identical to champion, SKIP"); continue
    ge=df.iloc[list(t-bt)]
    print(f"  {name:16} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f} f1 {ge.f1.median():>6,.0f} f11 {ge.f11.median():.4f}")

# build the ones worth uploading (novelty 2-12%, single clean lever)
BUILD = ["m028", "with_collcall", "veteran_boost", "spend_light"]
for name in BUILD:
    v=probes[name]; t=top_set(v); nov=100-overlap(t,bt)
    if 1.0 <= nov <= 15:
        sb.write_sub(v, ids, C.OUT/f"sub14_{name}.xlsx", f"s14_{name}")
        sb.framework_guard(C.OUT/f"sub14_{name}.xlsx")
    else:
        print(f"  {name}: novelty {nov:.1f}% out of range — not built")
print("\nAudit each built file, then upload the 2-3 with cleanest personas + highest ceiling.")