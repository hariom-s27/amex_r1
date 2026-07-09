"""experiments/batch_spend_scan.py — scan the SPEND coefficient. Research says spend is
near break-even after rewards, so 0.025 may be too high. Test lower/zero/negative."""
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
BEN=(35*df.f13.fillna(0)+df.f14.fillna(0)+17.5*df.f15.fillna(0)+df.f16.fillna(0)).values

# CHAMPION BASIS = risk_up_15 (0.922): block-excluded, m=0.35, lgd=1.5, ben=-1.0
def model(spend_w=0.025, m=0.35, lgd=1.5, ben=-1.0):
    v = spend_w*sp + m*f1 + ben*BEN - lgd*f11*(f1 + sp/12)
    return v - 1e9*catmiss

base = model(); bt = top_set(base)
champ_v = pd.ExcelFile(C.OUT/"sub15_risk_up_15.xlsx").parse("Predictions")["Prediction"].values
assert overlap(bt, top_set(champ_v)) > 99, "basis must match risk_up_15 champion (0.922)"

# scan spend weight DOWN (research: spend is break-even after 5x rewards)
probes = {
    "spend_018": model(spend_w=0.018),
    "spend_012": model(spend_w=0.012),
    "spend_008": model(spend_w=0.008),
    "spend_000": model(spend_w=0.000),   # spend contributes nothing (pure revolve/risk)
}
print("SPEND-SCAN PROFILES (novelty vs 0.922 champion):")
built=[]
for name,v in probes.items():
    t=top_set(v); nov=100-overlap(t,bt)
    ge=df.iloc[list(t-bt)]
    print(f"  {name:12} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f} "
          f"f1 {ge.f1.median():>6,.0f} f11 {ge.f11.median():.4f}")
    if 1.0 <= nov <= 20:
        sb.write_sub(v, ids, C.OUT/f"sub16_{name}.xlsx", f"s16_{name}")
        sb.framework_guard(C.OUT/f"sub16_{name}.xlsx")
        built.append(name)
print(f"\nbuilt: {built}")
print("audit each, upload spend_012 FIRST (moderate reduction, research-backed direction)")