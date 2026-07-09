"""experiments/gentle_probe.py — segment refinement done SANELY (merge by profit, not identity)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, imputed_spend, sink_spend
from src.scoring import prep, build, top_set, overlap
from src.predictor import load_anchors, make_predictor

df = load_data(); arr = prep(df); spend = imputed_spend(df)
files, y, tops = load_anchors(); pred = make_predictor(y, tops)
R = lambda x: pd.Series(np.asarray(x, float)).rank(pct=True).values
base = build({"m": 0.22, "lgd": 0.85}, spend, arr); champ = top_set(base)

def profile(name, score):
    t = top_set(score); nov = 100 - overlap(t, champ)
    err = sum((overlap(t, tp) - s) ** 2 for tp, s in zip(tops, y))
    enter = list(t - champ)
    ge = df.iloc[enter]
    print(f"{name:34} nov {nov:4.1f}% | backfit {err:6.1f} | "
          f"ENTER rev {100*(ge.f1>0).mean():3.0f}% catsum {ge.catsum.median():>7,.0f}")

print("SANE PROBES (want backfit < 20 AND novelty 10-20%):\n")
profile("champion (reference)", base)

# Idea 1: blend champion with a SMALL segment adjustment (not a takeover)
revolver = (df.f1 > 0).values
seg_adj = R(base).copy()
seg_adj[revolver] += 0.05      # tiny revolver bonus, ranking stays mostly intact
profile("champion + 5% revolver bonus", seg_adj)

# Idea 2: re-rank ONLY the imputed 23%, keep everyone else exactly as champion
catmiss = df[C.CATS].isna().all(axis=1).values
alt = base.copy()
alt[catmiss] = np.quantile(base, 0.10)   # push imputed members low (test: are they overranked?)
profile("imputed members ranked low", alt)

# Idea 3: imputed members ranked HIGH instead
alt2 = base.copy()
alt2[catmiss] = np.quantile(base, 0.60)
profile("imputed members ranked high", alt2)

# Idea 4: gentle utilization refinement (small weight, only where f17 exists)
util = np.where(df.f17.notna().values, df.f1.fillna(0).values/(df.f17.fillna(1).values+1), 0)
u = R(base) + 0.1 * R(util)
profile("champion + 10% utilization", u)

print("\nPick the SANE one (backfit<20, novelty 10-20%, personas make sense) for submission #5.")