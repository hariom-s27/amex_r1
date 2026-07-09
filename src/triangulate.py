"""triangulate.py — the ring engine: grid -> per-anchor err -> identification -> consensus."""
import itertools
import numpy as np
import pandas as pd
from . import config as C
from .scoring import build, top_set
from .predictor import load_anchors


def run_ring(grid_dict, spend, arr, noise=0.8, top_n=15):
    """
    grid_dict: {knob: [values]}. Returns sorted (err, cfg) list + prints tables.
    Anchors are read from anchors.json (REAL scores only).
    """
    files, y, tops = load_anchors()
    A = list(zip(tops, y))
    keys = list(grid_dict)
    combos = list(itertools.product(*[grid_dict[k] for k in keys]))
    print(f"{len(combos)} configs across {files}")

    res = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        t = top_set(build(cfg, spend, arr))
        err = sum((100 * len(t & mm) / C.K - s) ** 2 for mm, s in A)
        res.append((err, cfg))
    res.sort(key=lambda x: x[0])

    print(f"\nTOP {top_n} (err | cfg):")
    for e, c in res[:top_n]:
        print(f"  {e:6.2f}  {c}")

    print("\nIDENTIFICATION TABLE (flat row = knob invisible to real anchors):")
    for kk in keys:
        best = {}
        for e, c in res:
            best[c[kk]] = min(best.get(c[kk], 1e9), e)
        print(f"  {kk}: " + "  ".join(f"{v}={best[v]:.1f}" for v in sorted(best, key=str)))
    return res


def consensus(res, spend, arr, noise=1.0, max_families=200):
    """Vote across all distinct configs fitting within `noise`. Returns a score array."""
    fams = []
    for e, c in res:
        if e > noise or len(fams) >= max_families:
            break
        t = frozenset(top_set(build(c, spend, arr)))
        if all(len(t & f) / C.K < 0.999 for f in fams):
            fams.append(t)
    print(f"consensus over {len(fams)} distinct families")
    counts = np.zeros(C.N, np.int32)
    for t in fams:
        counts[list(t)] += 1
    tie = pd.Series(build(res[0][1], spend, arr)).rank().values
    return counts.astype(float) * 1e6 + tie, fams