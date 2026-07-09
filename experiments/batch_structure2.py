"""experiments/batch_structure.py — coefficients are maxed; test STRUCTURE.
Unbundle benefits (real product-sheet weights) + spend×revolve interaction."""
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

# CHAMPION BASIS = risk_up_15 (0.922): m=0.35, lgd=1.5, ben bundle -1.0, spend 0.025
def base_model(ben_term):
    v = 0.025*sp + 0.35*f1 - ben_term - 1.5*f11*(f1 + sp/12)
    return v - 1e9*catmiss

BEN_STD = 35*f13 + f14 + 17.5*f15 + f16   # current bundle
base = base_model(1.0*BEN_STD); bt = top_set(base)
champ_v = pd.ExcelFile(C.OUT/"sub15_risk_up_15.xlsx").parse("Predictions")["Prediction"].values
assert overlap(bt, top_set(champ_v)) > 99, "basis must match 0.922 champion"

probes = {
    # unbundle: heavier lounge (premium benefit), lighter cab
    "ben_lounge_hi": base_model(50*f13 + f14 + 10*f15 + f16),
    "ben_cab_hi":    base_model(20*f13 + f14 + 40*f15 + f16),
    "ben_equal":     base_model(20*f13 + 20*f14 + 20*f15 + 20*f16),
    "ben_credits_hi":base_model(35*f13 + 2*f14 + 17.5*f15 + 2*f16),  # $ credits weighted higher
    # spend x revolve interaction (active revolvers most profitable)
    "spend_x_revolve": (0.025*sp + 0.35*f1 + 0.00000005*sp*f1
                        - 1.0*BEN_STD - 1.5*f11*(f1+sp/12) - 1e9*catmiss),
}
print("STRUCTURE PROBES (novelty vs 0.922 champion):")
built=[]
for name,v in probes.items():
    t=top_set(v); nov=100-overlap(t,bt)
    ge=df.iloc[list(t-bt)]
    print(f"  {name:16} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f} "
          f"f1 {ge.f1.median():>6,.0f} benefit {(f13+f15)[list(t-bt)].mean():.1f}")
    if 1.0 <= nov <= 20:
        sb.write_sub(v, ids, C.OUT/f"sub17_{name}.xlsx", f"s17_{name}")
        sb.framework_guard(C.OUT/f"sub17_{name}.xlsx")
        built.append(name)
print(f"\nbuilt: {built}")
print("audit each; upload the one with most sensible ENTER persona first")