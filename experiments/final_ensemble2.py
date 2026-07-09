"""experiments/final_ensemble.py — the ONE protective submission. Build, verify, done."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.scoring import top_set, overlap
from src.submit import write_sub, framework_guard

R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
ids = pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['ID'].values
# our BEST scored files, weighted by their real scores
W = {"sub04_impute2.xlsx":0.875, "sub07_risk_moderate.xlsx":0.842, "sub06_final.xlsx":0.834}
V = {f: pd.ExcelFile(C.OUT/f).parse("Predictions")["Prediction"].values for f in W}
champ = top_set(V["sub04_impute2.xlsx"])

# champion-dominant rank-mean (protects the proven ranking, smooths the boundary)
ens = 0.6*R(V["sub04_impute2.xlsx"]) + 0.25*R(V["sub07_risk_moderate.xlsx"]) + 0.15*R(V["sub06_final.xlsx"])
t = top_set(ens); nov = 100-overlap(t,champ)
print(f"final ensemble: novelty vs champion {nov:.1f}%  (want 3-8% — smooths boundary, keeps champion)")
ge = df.iloc[list(t-champ)] if 'df' in dir() else None
print(f"boundary members swapped: {int(nov/100*C.K):,}")

if 3 <= nov <= 10:
    write_sub(ens, ids, C.OUT/"sub10_final_ensemble.xlsx","s10_final_ensemble")
    framework_guard(C.OUT/"sub10_final_ensemble.xlsx")
    print("BUILT sub10_final_ensemble — this is the protective submission.")
else:
    print(f"novelty {nov:.1f}% out of range — tell Claude before building.")