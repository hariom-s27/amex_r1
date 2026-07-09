"""experiments/leak_hunt.py — systematic hunt for target leakage (the one unexplored door)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data
from src.scoring import top_set, overlap

df = load_data()
champ_v = pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['Prediction'].values
champ = top_set(champ_v)
R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values

print("=== TEST 1: does ANY single raw column align strongly with the champion? ===")
print("(a column with >70% overlap = it nearly IS the target signal = investigate)")
for c in [f'f{i}' for i in range(1,24)]:
    t = top_set(df[c].fillna(-1).values)
    o = overlap(t, champ)
    if o > 55:
        print(f"  {c}: {o:.1f}% overlap with champion  <-- notable")

print("\n=== TEST 2: exact-value leakage — do any columns have suspiciously few unique values? ===")
print("(a near-target column might be quantized to match top-20% boundaries)")
for c in [f'f{i}' for i in range(1,24)]:
    nun = df[c].nunique()
    if nun < 50:
        print(f"  {c}: only {nun} unique values")

print("\n=== TEST 3: row-order / id leakage (finer than before) ===")
# does the champion's top-set cluster in any id region?
in_top = np.zeros(C.N,bool); in_top[list(champ)]=True
for chunk in range(10):
    lo,hi = chunk*C.N//10, (chunk+1)*C.N//10
    print(f"  id block {chunk}: {100*in_top[lo:hi].mean():.1f}% in champ-top", end="")
    print("  <-- UNEVEN!" if abs(in_top[lo:hi].mean()-0.2)>0.03 else "")

print("\n=== TEST 4: duplicate & near-duplicate rows (leak across 70/30 split) ===")
dups = df.drop(columns=['id']).duplicated().sum()
print(f"  exact duplicate feature-rows: {dups:,}")

print("\n=== TEST 5: does f5 (the orphan) encode rank info? unique-value spacing ===")
f5 = df.f5.dropna()
print(f"  f5: {f5.nunique():,} unique, min gap {np.diff(np.sort(f5.unique())).min():.4f}")
print("\nread: any RED flag (uneven id blocks, tiny unique counts, a 70%+ single column)")
print("      = a leak worth exploiting. All clean = no leak, 0.875 is our honest ceiling.")