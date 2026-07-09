"""experiments/probes.py — build both structural probes, show profiles (not just scores)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, build, top_set, overlap
from src.predictor import load_anchors, make_predictor

df = load_data(); arr = prep(df); spend = imputed_spend(df)
files, y, tops = load_anchors(); pred = make_predictor(y, tops)
R = lambda x: pd.Series(np.asarray(x, float)).rank(pct=True).values

base = build({"m": 0.22, "lgd": 0.85}, spend, arr)
champ = top_set(base)
f1 = df.f1.fillna(0).values; f17 = df.f17.fillna(0).values; f11 = df.f11.fillna(0).values

# ---- PROBE 1: segment ranking (revolvers vs transactors ranked separately) ----
revolver = f1 > 0
seg = np.empty(C.N)
seg[revolver]  = R(base[revolver])        # rank within revolvers
seg[~revolver] = R(base[~revolver])       # rank within transactors
# merge: give revolvers the top band, transactors below (test the structure)
seg_score = np.where(revolver, 0.5 + 0.5 * seg, 0.5 * seg)

# ---- PROBE 2: utilization-led (f1/f17 as a PRIMARY term) ----
util = f1 / (f17 + 1)
util_score = 0.5 * R(base) + 0.5 * R(util)      # heavy, co-equal weight

def profile(name, score):
    t = top_set(score)
    nov = 100 - overlap(t, champ)
    err = sum((overlap(t, tp) - s) ** 2 for tp, s in zip(tops, y))
    enter = list(t - champ); leave = list(champ - t)
    ge = df.iloc[enter]; gl = df.iloc[leave]
    print(f"\n=== {name} ===")
    print(f"  novelty vs champion: {nov:.1f}%   (want 20-40% for a real probe)")
    print(f"  backfit-err vs anchors: {err:.1f}   (moderate=alt model, huge=broken)")
    print(f"  predicted (UNTRUSTED, for ref): {pred(t):.1f}")
    print(f"  ENTER n={len(enter):,}: revolver {100*(ge.f1>0).mean():.0f}% | "
          f"med catsum {ge.catsum.median():,.0f} | med f11 {ge.f11.median():.4f}")
    print(f"  LEAVE n={len(leave):,}: revolver {100*(gl.f1>0).mean():.0f}% | "
          f"med catsum {gl.catsum.median():,.0f} | med f11 {gl.f11.median():.4f}")

profile("PROBE 1 — segment ranking", seg_score)
profile("PROBE 2 — utilization-led", util_score)
print("\nDECISION: pick the probe with 20-40% novelty AND sensible personas.")
print("If one has huge backfit-err (>50) it's likely broken, not bold — tell Claude.")