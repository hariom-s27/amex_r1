"""experiments/run.py — setup + unused-variable screen v2."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, sink_spend, imputed_spend
from src.scoring import prep, build, top_set
from src.predictor import load_anchors, self_test, make_predictor

# --- setup ---
df = load_data()
arr = prep(df)
spend = imputed_spend(df)

files, y, tops = load_anchors()
self_test(files, y, tops)
pred = make_predictor(y, tops)

# --- unused-variable screen v2: sweep REAL weight ---
R = lambda x: pd.Series(x).rank(pct=True).values
base = build({"m": 0.22, "lgd": 0.85}, spend, arr)
base_r = pd.Series(base).rank(pct=True).values
print("\nUNUSED-VARIABLE SCREEN v2 (best predicted | novelty at best weight):")
for col in ["f12", "f19", "f20", "f22", "f4", "f11", "f5"]:
    cv = R(df[col].fillna(0).values)
    best = (-1, None, None)
    for w in [0.05, 0.1, 0.2, 0.35, 0.5]:
        cand = (1 - w) * base_r + w * cv
        t = top_set(cand)
        p = pred(t)
        nov = 100 - 100 * len(t & top_set(base)) / C.K
        if p > best[0]:
            best = (p, w, nov)
    print(f"  +{col}: best pred {best[0]:5.1f} at weight {best[1]} | novelty {best[2]:4.1f}%")
print("\nread: pred >= 84 with novelty 8-25% = real candidate. f11/f5 are controls (known).")