"""experiments/ensemble_study.py — will a rank-mean ensemble actually help, or collapse to champion?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.scoring import top_set, overlap

R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
files = {"champ_875":"sub04_impute2.xlsx","risk_842":"sub07_risk_moderate.xlsx",
         "f5_834":"sub06_final.xlsx","engine_820":"FINAL_submission_r1.xlsx"}
V = {n: pd.ExcelFile(C.OUT/f).parse("Predictions")["Prediction"].values for n,f in files.items()}
champ = top_set(V["champ_875"])

print("=== how different are our scored files? ===")
ns=list(V)
for i,a in enumerate(ns):
    for b in ns[i+1:]:
        print(f"  {a} ~ {b}: {overlap(top_set(V[a]),top_set(V[b])):.1f}%")

print("\n=== ensemble candidates (rank-mean, various weights) ===")
ensembles = {
    "E1: champ only (ref)":        R(V["champ_875"]),
    "E2: 0.7champ+0.3risk":        0.7*R(V["champ_875"])+0.3*R(V["risk_842"]),
    "E3: 0.6champ+0.2risk+0.2f5":  0.6*R(V["champ_875"])+0.2*R(V["risk_842"])+0.2*R(V["f5_834"]),
    "E4: equal 4-way":             sum(R(v) for v in V.values())/4,
    "E5: 0.5champ+0.5(3-way avg)": 0.5*R(V["champ_875"])+0.5*(R(V["risk_842"])+R(V["f5_834"])+R(V["engine_820"]))/3,
}
for n,e in ensembles.items():
    t=top_set(e); nov=100-overlap(t,champ)
    print(f"  {n:30} novelty vs champ {nov:4.1f}%")
print("\nread: novelty 3-10% = ensemble smooths the boundary (good, stable).")
print("      novelty >15% = ensemble drifted too far from proven champion (risky).")
print("      novelty ~0% = collapses to champion (safe but pointless).")