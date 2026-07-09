"""experiments/interactions.py — test ratios & interactions through the validated predictor."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, build, top_set
from src.predictor import load_anchors, make_predictor

df = load_data(); arr = prep(df); spend = imputed_spend(df)
files, y, tops = load_anchors(); pred = make_predictor(y, tops)
R = lambda x: pd.Series(np.asarray(x, float)).rank(pct=True).values

base = build({"m": 0.22, "lgd": 0.85}, spend, arr)
base_r = R(base); champ = top_set(base)

f1 = df.f1.fillna(0).values; f17 = df.f17.fillna(0).values
f19 = df.f19.fillna(1).values; f11 = df.f11.fillna(0).values
cat = df.catsum.fillna(0).values

cands = {
    "utilization f1/f17":   f1 / (f17 + 1),
    "spend_per_acct":       cat / (f19 + 1),
    "spend x (1-risk)":     R(spend) * (1 - R(f11)),
    "spend x revolve":      R(spend) * R(f1),
    "risk_adj_spend":       R(spend) - R(f11),
    "spend / (1+risk)":     R(spend) / (1 + R(f11)),
    "log spend x acct":     np.log1p(spend) * R(f19),
    "revolve x low-risk":   R(f1) * (1 - R(f11)),
}
print("INTERACTION SCREEN (best pred | novelty):")
for name, raw in cands.items():
    cv = R(raw); best = (-1, None, None)
    for w in [0.1, 0.2, 0.35, 0.5, 0.7]:
        cand = (1 - w) * base_r + w * cv
        t = top_set(cand); p = pred(t)
        nov = 100 - 100 * len(t & champ) / C.K
        if p > best[0]: best = (p, w, nov)
    print(f"  {name:22} pred {best[0]:5.1f} at w={best[1]} | novelty {best[2]:4.1f}%")
print("\npred>=84 & novelty 8-25% = build it for submission #5. All flat = go to segment ranking.")