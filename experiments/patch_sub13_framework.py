# experiments/patch_sub13_framework.py
import pandas as pd
from pathlib import Path
FILE = Path("outputs/submissions/sub13b_evict_only.xlsx")   # or sub13a_ben_evict.xlsx
xl = pd.ExcelFile(FILE); pred = xl.parse("Predictions"); fw = xl.parse("Profitability Framework")
fw.loc[fw.Section=="Feature Transformations","Response"] = (
 "No scaling (data pre-winsorised). Reported spend clipped at 0; no imputation of unreported "
 "spend. Members with no reported category-spend block are additionally excluded from the "
 "top-quintile selection: with no observable billed volume, top-quintile profitability cannot "
 "be demonstrated, so they rank below all reported-spend members (conservative issuer stance).")
fw.loc[fw.Section=="Business Logic","Response"] = (
 "Issuer books interchange on observed volume and interest on revolve; absent any reported "
 "billed volume, a member's profitability claim rests on lending economics alone, which this "
 "framework treats as insufficient for top-quintile classification.")
with pd.ExcelWriter(FILE, engine="openpyxl") as w:
    pred.to_excel(w, sheet_name="Predictions", index=False)
    fw.to_excel(w, sheet_name="Profitability Framework", index=False)
print("patched:", FILE, "| rerun: python -m src.audit_submission", FILE)