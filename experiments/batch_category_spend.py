"""experiments/batch_category_spend.py — the untested structure: individual category
spends f6-f10 with different net margins (tiered rewards: 5x travel costs more)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, sink_spend
from src.scoring import top_set, overlap
import src.submit as sb

df = load_data(); ids = df["id"].values
catmiss = df[C.CATS].isna().all(axis=1).values
f1=df.f1.fillna(0).values; f11=df.f11.fillna(0).values
f13=df.f13.fillna(0).values; f14=df.f14.fillna(0).values
f15=df.f15.fillna(0).values; f16=df.f16.fillna(0).values
# individual category spends (clip negatives = refunds)
f6=df.f6.clip(lower=0).fillna(0).values   # possibly travel (5x rewards)
f7=df.f7.clip(lower=0).fillna(0).values   # other spend
f8=df.f8.clip(lower=0).fillna(0).values
f9=df.f9.clip(lower=0).fillna(0).values
f10=df.f10.clip(lower=0).fillna(0).values
catsum = f6+f7+f8+f9+f10
BEN = 35*f13 + f14 + 40*f15 + f16
def model(w6,w7,w8,w9,w10):
    spend = w6*f6 + w7*f7 + w8*f8 + w9*f9 + w10*f10
    v = 0.025*spend + 0.35*f1 - 1.0*BEN - 1.5*f11*(f1 + spend/12)
    return v - 1e9*catmiss

base = model(1,1,1,1,1); bt = top_set(base)   # equal = current champion
champ_v = pd.ExcelFile(C.OUT/"sub17_ben_cab_hi.xlsx").parse("Predictions")["Prediction"].values
print(f"equal-weight base vs champion: {overlap(bt,top_set(champ_v)):.1f}% (should be ~100)")

# probe different category margins (tiered rewards -> some categories net less)
probes = {
    "f6_low":   model(0.5,1,1,1,1),   # if f6=travel (5x rewards), net margin lower
    "f6_high":  model(1.5,1,1,1,1),
    "f7_high":  model(1,1.5,1,1,1),   # if f7=other (1x), net margin higher
    "f7_low":   model(1,0.5,1,1,1),
    "travel_discount": model(0.6,1.2,1,1,0.6),  # 5x categories netted down
}
print("\nCATEGORY-SPEND PROBES (novelty vs champion):")
built=[]
for name,v in probes.items():
    t=top_set(v); nov=100-overlap(t,bt)
    ge=df.iloc[list(t-bt)] if len(t-bt)>0 else df.iloc[:1]
    print(f"  {name:16} novelty {nov:4.1f}% | ENTER catsum {ge[list(C.CATS)].sum(axis=1).median():>7,.0f}")
    if 1.0 <= nov <= 20:
        sb.write_sub(v, ids, C.OUT/f"sub21_{name}.xlsx", f"s21_{name}")
        sb.framework_guard(C.OUT/f"sub21_{name}.xlsx")
        built.append(name)
print(f"\nbuilt: {built}")
print("If any category re-weight has real novelty + sensible persona = the tiered-rewards")
print("structure the leaders may use. This is the last untested STRUCTURE.")