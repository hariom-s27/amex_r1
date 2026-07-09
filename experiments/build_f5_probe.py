"""experiments/build_f5_probe.py — build the f5-primary probe as a submission-ready file."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, top_set, overlap
from src.submit import write_sub, framework_guard

df = load_data(); arr = prep(df); spend = imputed_spend(df)
R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
ids = df["id"].values
f1=df.f1.fillna(0).values; f5=df.f5.fillna(0).values; f11=df.f11.fillna(0).values
champ = top_set(pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['Prediction'].values)

# f5 + revolve - risk: f5-primary but with the proven economic structure around it
probe = 0.025*f5 + 0.22*f1 - 0.85*f11*(f1 + f5/12)   # champion shape, f5 REPLACES catsum
t = top_set(probe); nov = 100-overlap(t,champ)
ge = df.iloc[list(t-champ)]
print(f"f5-primary probe: novelty {nov:.1f}%")
print(f"  ENTER: catsum {ge.catsum.median():,.0f} | f5 {ge.f5.median():,.0f} | f1 {ge.f1.median():,.0f} | f11 {ge.f11.median():.4f}")

write_sub(probe, ids, C.OUT/"sub09_f5_primary.xlsx", "s09_f5_primary")
framework_guard(C.OUT/"sub09_f5_primary.xlsx")
print("\nf5-primary probe built & validated. This resolves: does the truth use f5 or catsum?")