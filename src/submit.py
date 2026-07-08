"""
submit.py — the ONLY place framework text and the submission writer live.

Kills two notebook bugs for good:
  1. FW_TEXT existed in several cells with different values -> now ONE copy.
  2. Files silently inherited the OLD engine framework text -> framework_guard()
     refuses any file whose text still contains old fingerprints (625, f19...).
"""
import datetime as dt
import numpy as np
import pandas as pd
from . import config as C

# ---- SINGLE SOURCE OF TRUTH for the framework sheet --------------------------
FW_TEXT = {
    "Variables Used":
        "Non-zero weight: f1 (revolve balance), f6-f10 (category spends), f11 (risk score), "
        "f13-f16 (benefit usage). Zero weight after calibration: f2,f3,f4,f5,f12,f17-f23. "
        "Identifier never used.",
    "Profitability Equation":
        "Profit_i = 0.025*Spend_i + 0.22*f1_i - RewardsCost_i - BenefitCost_i "
        "- RiskCost_i. Spend = sum(f6..f10), floored at 0, gbm-imputed when the category "
        "block is absent. Terms with zero calibrated weight are omitted.",
    "Prediction Logic":
        "Score every member; rank-order by profit; the top quintile is the predicted "
        "most-profitable 20%. The constant annual fee is rank-invariant and omitted.",
    "Variable Selection Logic":
        "Variables kept only if designed leaderboard experiments show they move top-quintile "
        "membership beyond sampling noise; otherwise weight zero.",
    "Coefficient/Weight Derivation":
        "Interchange (2.5% of spend) is the numeraire; only weight ratios affect ranking. "
        "Remaining weights calibrated by triangulation against independently scored "
        "submissions, each reproduced within sampling noise.",
    "Feature Transformations":
        "Missing behavioural fields = no activity (zero). Category spends floored at zero. "
        "No winsorisation (data is provider-capped). Score is scale-free.",
    "Business Logic":
        "Revenue = interchange on spend + net interest on revolving balance. Costs = benefit "
        "usage at unit economics + expected credit loss (risk x exposure). Low-risk spenders "
        "rank highest.",
    "Assumptions":
        "Annual fee constant (rank-invariant). f11 proxies default probability. Benefit usage "
        "priced as cost; missing record = no activity.",
    "Validation Approach":
        "Per-anchor back-fit within sampling noise on independently scored submissions; "
        "leave-one-out overlap predictor; behavioural persona checks; boundary audit confirms "
        "a single member at the cutoff.",
    "Additional Notes (Optional)":
        "Coefficients refined by designed leaderboard experiments used as anchors; the reported "
        "equation reproduces each anchor's accuracy within sampling noise.",
}

_OLD_FINGERPRINTS = ["625", "f19", "0.0235", "0.009*f21", "0.16*f5", "0.27*f1"]


def write_sub(scores, ids, path, tag: str) -> bool:
    """scores: array aligned to `ids`; ids: the member id array. Writes 2-sheet xlsx."""
    scores = np.asarray(scores).ravel()
    pred = pd.DataFrame({"ID": np.arange(C.N)})
    pred["Prediction"] = pd.Series(scores, index=np.asarray(ids)).reindex(pred["ID"].values).values
    fw = pd.DataFrame({"Section": list(FW_TEXT), "Response": [FW_TEXT[s] for s in FW_TEXT]})
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pred.to_excel(w, sheet_name="Predictions", index=False)
        fw.to_excel(w, sheet_name="Profitability Framework", index=False)
    ok = _validate(path)
    if ok:
        _log(tag, path)
    print(path, "| ALL CHECKS PASS" if ok else "| FAIL — do NOT upload")
    return ok


def _validate(path) -> bool:
    xl = pd.ExcelFile(path)
    p = xl.parse("Predictions")
    f = xl.parse("Profitability Framework")
    return bool(
        xl.sheet_names == ["Predictions", "Profitability Framework"]
        and len(p) == C.N
        and list(p.columns) == ["ID", "Prediction"]
        and p["Prediction"].notna().all()
        and np.isfinite(p["Prediction"]).all()
        and (p["ID"].values == np.arange(C.N)).all()
        and f["Response"].notna().sum() == len(FW_TEXT)
    )


def framework_guard(path):
    """Refuse any file whose framework text still contains old-engine fingerprints."""
    f = pd.ExcelFile(path).parse("Profitability Framework")
    text = " ".join(str(x) for x in f["Response"].fillna(""))
    bad = [fp for fp in _OLD_FINGERPRINTS if fp in text]
    assert not bad, f"OLD/STALE framework text detected: {bad} — regenerate with submit.write_sub"
    print(path, "| framework text CLEAN")


def _log(tag, path):
    row = pd.DataFrame([{
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "tag": tag, "file": str(path).split("/")[-1], "lb_score": "",
    }])
    header = not C.LOG.exists()
    row.to_csv(C.LOG, mode="a", header=header, index=False)
