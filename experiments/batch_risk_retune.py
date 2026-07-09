"""batch_risk_retune.py — m is plateauing at 0.921; test RISK re-tuning on the m=0.35 base.
Heavy revolve weight means risk penalty may need to rise. Each probe = one clean lever."""
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

# CHAMPION BASIS = m035 (0.921): block-excluded, m=0.35, lgd=0.85, ben=-1.0
def model(m=0.35, lgd=0.85, ben=-1.0):
    v = 0.025*sp + m*f1 + ben*BEN - lgd*f11*(f1 + sp/12)
    return v - 1e9*catmiss

base = model(); bt = top_set(base)
champ_v = pd.ExcelFile(C.OUT/"sub14_m035.xlsx").parse("Predictions")["Prediction"].values
assert overlap(bt, top_set(champ_v)) > 99, "basis must match m035 champion"

# probes: risk re-tune (newly live after heavier revolve) + one m-plateau confirm
probes = {
    "m037":        model(m=0.37),                    # confirm plateau top
    "risk_up_15":  model(lgd=1.5),                   # heavier revolve -> heavier risk?
    "risk_up_25":  model(lgd=2.5),
    "risk_dn_05":  model(lgd=0.5),                   # or lighter?
    "ben_light":   model(ben=-0.5),                  # benefit magnitude scan
    "ben_heavy":   model(ben=-1.5),
}

print("PROBE PROFILES (novelty vs m035 champion 0.921):")
built = []
for name, v in probes.items():
    t = top_set(v); nov = 100 - overlap(t, bt)
    if nov < 0.5:
        print(f"  {name:12} novelty {nov:.1f}%  <-- identical, SKIP"); continue
    ge = df.iloc[list(t-bt)]
    print(f"  {name:12} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f} "
          f"f1 {ge.f1.median():>6,.0f} f11 {ge.f11.median():.4f}")
    if 1.0 <= nov <= 15:
        sb.write_sub(v, ids, C.OUT/f"sub15_{name}.xlsx", f"s15_{name}")
        sb.framework_guard(C.OUT/f"sub15_{name}.xlsx")
        built.append(name)
print(f"\nbuilt: {built}")
print("audit each, then upload risk_up_15 FIRST (highest-value: risk axis newly live after m-climb)")