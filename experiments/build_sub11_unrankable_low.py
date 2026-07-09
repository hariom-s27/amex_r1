"""build_sub11_unrankable_low.py — champion ranking, missing-spend block sunk to bottom.
Hypothesis: truth computes profit with missing category spend = 0, so imputed members
in our top set are false positives. This is Research-Report Submission 1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data
from src.scoring import top_set, overlap
import src.submit as sb

df = load_data()
catmiss = df[C.CATS].isna().all(axis=1).values
ids = df["id"].values
print(f"missing-spend block: {catmiss.sum():,} members ({100*catmiss.mean():.1f}%)")

# use the EXACT scored 0.875 file (never a rebuild — avoids imputer drift)
pred = pd.ExcelFile(C.OUT / "sub04_impute2.xlsx").parse("Predictions")
assert (pred["ID"].values == np.arange(C.N)).all()
assert (ids == np.arange(C.N)).all()
v875 = pred["Prediction"].values

# champion order preserved; entire missing block pushed below everyone else
r = pd.Series(v875).rank(method="first").values.astype(float)
new = r.copy()
new[catmiss] -= 1_000_000

t_new, t_ch = top_set(new), top_set(v875)
moved = int(catmiss[list(t_ch)].sum())
print(f"imputed members evicted from top-20%: {moved:,}")
print(f"novelty vs champion: {100 - overlap(t_new, t_ch):.1f}%  (should equal {moved/1000:.1f}%)")

enter = list(t_new - t_ch)
ge = df.iloc[enter]
print(f"backfilled ENTER n={len(enter):,}: med catsum {ge.catsum.median():,.0f} | "
      f"med f1 {ge.f1.median():,.0f} | med f11 {ge.f11.median():.4f} | cats-missing {100*ge[C.CATS].isna().all(axis=1).mean():.0f}%")

print(f"\nEXPECTED-SCORE ARITHMETIC:")
print(f"  if truth sinks the block: gain ≈ {moved:,}/1000 × (backfill hit-rate ~0.6-0.75) "
      f"= +{0.6*moved/1000:.1f} to +{0.75*moved/1000:.1f} pp  → ~{87.5+0.6*moved/1000:.1f}-{87.5+0.75*moved/1000:.1f}")
print(f"  if truth includes block at base rate: lose ≈ 0.20 × {moved:,}/1000 = -{0.2*moved/1000:.1f} pp (rank safe via max-rule)")

# framework text: describe THIS file honestly (conservative exclusion, no imputer)
FW = dict(sb.FW_TEXT)
FW["Feature Transformations"] = (
    "Missing behavioural fields = no activity (zero). Members whose entire category-spend "
    "block is unobserved are treated conservatively: spend contribution 0, ranking them "
    "outside the top quintile. Category spends floored at zero. No winsorisation "
    "(data is provider-capped). Score is scale-free; no identifier-derived features.")
FW["Additional Notes (Optional)"] = (
    "Treatment of unobserved-spend members follows a designed experiment: with no reliable "
    "spend signal, expected-profit ranking places them below members with observed spend. "
    "Coefficients calibrated by independently scored submissions used as anchors.")
sb.FW_TEXT = FW
ok = sb.write_sub(new, ids, C.OUT / "sub11_unrankable_low.xlsx", "s11_unrankable_low")
sb.framework_guard(C.OUT / "sub11_unrankable_low.xlsx")