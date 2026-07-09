"""audit.py — pre-flight checks before spending a submission slot."""
import numpy as np
import pandas as pd
from . import config as C
from .scoring import top_set
from .predictor import load_anchors, make_predictor


def preflight(scores, champion_name="sub04_impute2.xlsx"):
    """Run all pre-submission audits on a candidate score array."""
    files, y, tops = load_anchors()
    pred = make_predictor(y, tops)
    t = top_set(scores)

    # backfit vs every real anchor
    print("BACKFIT vs real anchors:")
    err = 0
    for f, s, tp in zip(files, y, tops):
        o = 100 * len(t & tp) / C.K
        err += (o - s) ** 2
        print(f"  {f:26} real {s:5.1f}  overlap {o:5.1f}  resid {o - s:+.1f}")
    print(f"  backfit-err = {err:.2f}  (<2 excellent, 2-6 ok, >6 REJECT)")

    # predicted score (conservative floor)
    print(f"\nPREDICTED score (floor, +~4pp real): {pred(t):.1f}")

    # novelty vs champion
    champ_path = C.OUT / champion_name
    if champ_path.exists():
        cv = pd.ExcelFile(champ_path).parse("Predictions")["Prediction"].values
        nov = 100 - 100 * len(t & top_set(cv)) / C.K
        print(f"novelty vs champion: {nov:.1f}%  (10-25% = useful new bet)")

    # boundary fragility
    v = np.asarray(scores, float)
    cut = np.quantile(v, 0.8)
    span = np.quantile(v, 0.99) - np.quantile(v, 0.01)
    near = int((np.abs(v - cut) <= 0.005 * span).sum())
    print(f"boundary members within +/-0.5% of cutoff: {near:,} (fewer = stabler)")

    verdict = "READY" if err < 6 else "REJECT — backfit too high"
    print(f"\nVERDICT: {verdict}")
    return err