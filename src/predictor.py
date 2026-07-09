"""predictor.py — the LOO-validated score predictor (your one trustworthy tool)."""
import json
import numpy as np
import pandas as pd
from . import config as C


def load_anchors():
    """Read real scores from anchors.json + their top-sets from disk."""
    raw = json.load(open(C.ANCHORS_JSON))
    files, scores, tops = [], [], []
    for name, s in raw.items():
        if name.startswith("_"):
            continue
        path = C.OUT / name
        if not path.exists():
            print(f"  [warn] {name} not found in outputs/submissions — skipped")
            continue
        v = pd.ExcelFile(path).parse("Predictions")["Prediction"].values
        files.append(name)
        scores.append(float(s))
        tops.append(set(np.argpartition(-v, C.K)[:C.K]))
    return files, np.array(scores), tops


def _matrix(tops):
    n = len(tops)
    return np.array([[100 * len(tops[i] & tops[j]) / C.K for j in range(n)] for i in range(n)])


def self_test(files, y, tops):
    """Each anchor predicted from the others. Close = tool trustworthy."""
    G = _matrix(tops)
    print("TOOL SELF-TEST (LOO):")
    for i in range(len(y)):
        idx = [j for j in range(len(y)) if j != i]
        ai = np.linalg.solve(G[np.ix_(idx, idx)] - 20, y[idx] - 20)
        q = np.array([100 * len(tops[i] & tops[j]) / C.K for j in idx])
        print(f"  {files[i]:26} real {y[i]:5.1f}  predicted {20 + (q - 20) @ ai:5.1f}")


def make_predictor(y, tops):
    G = _matrix(tops)
    a_full = np.linalg.solve(G - 20, y - 20)

    def pred(candidate_set):
        q = np.array([100 * len(candidate_set & t) / C.K for t in tops])
        return 20 + (q - 20) @ a_full
    return pred