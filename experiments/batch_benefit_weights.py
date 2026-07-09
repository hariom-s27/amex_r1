"""experiments/batch_benefit_weights.py — cab_hi proved benefit weights matter.
Scan each benefit weight to match the truth's calibration. This is the path to 0.929."""
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

# CHAMPION BASIS = ben_cab_hi (0.923): cab weight already raised to 40
def model(w13=35, w14=1, w15=40, w16=1):   # w15=40 is the new champion (cab_hi)
    ben = w13*f13 + w14*f14 + w15*f15 + w16*f16
    v = 0.025*sp + 0.35*f1 - 1.0*ben - 1.5*f11*(f1 + sp/12)
    return v - 1e9*catmiss

base = model(); bt = top_set(base)
champ_v = pd.ExcelFile(C.OUT/"sub17_ben_cab_hi.xlsx").parse("Predictions")["Prediction"].values
assert overlap(bt, top_set(champ_v)) > 99, "basis must match ben_cab_hi (0.923)"

# scan each benefit weight around its current value, one at a time
probes = {
    "cab_60":     model(w15=60),          # push cab further (it's working)
    "cab_80":     model(w15=80),
    "lounge_60":  model(w13=60),          # lounge higher
    "lounge_20":  model(w13=20),          # lounge lower
    "credits_hi": model(w14=5, w16=5),    # dollar-credits (f14,f16) higher
    "credits_0":  model(w14=0, w16=0),    # dollar-credits off
}
print("BENEFIT-WEIGHT SCAN (novelty vs 0.923 champion):")
built=[]
for name,v in probes.items():
    t=top_set(v); nov=100-overlap(t,bt)
    if nov < 0.5:
        print(f"  {name:12} novelty {nov:.1f}%  <-- identical, SKIP"); continue
    ge=df.iloc[list(t-bt)]
    print(f"  {name:12} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f} f1 {ge.f1.median():>6,.0f}")
    if 1.0 <= nov <= 15:
        sb.write_sub(v, ids, C.OUT/f"sub18_{name}.xlsx", f"s18_{name}")
        sb.framework_guard(C.OUT/f"sub18_{name}.xlsx")
        built.append(name)
print(f"\nbuilt: {built}")
print("audit each; upload cab_60 FIRST (cab weight is the proven-working direction)")