"""scoring.py — the ONE definition of build() and the overlap helpers."""
import numpy as np
import pandas as pd
from . import config as C


def prep(df):
    """Pull the raw arrays build() needs, once."""
    return {
        "f1": df.f1.fillna(0).values,
        "f11": df.f11.fillna(0).values,
        "f3": df.f3.fillna(0).values,
        "f5": df.f5.fillna(0).values,
        "f12": df.f12.fillna(0).values,
        "BEN": (35 * df.f13.fillna(0) + df.f14.fillna(0)
                + 17.5 * df.f15.fillna(0) + df.f16.fillna(0)).values,
    }


def build(cfg, spend, arr):
    """
    cfg keys: m, lgd, ben, w5, coll, w12 (all optional, default 0/off).
    spend: the spend array (sink or imputed). arr: output of prep().
    Returns a float score per member.
    """
    m = cfg.get("m", 0.22)
    lgd = cfg.get("lgd", 0.85)
    ben = cfg.get("ben", -1.0)
    w5 = cfg.get("w5", 0.0)
    coll = cfg.get("coll", 0.0)
    w12 = cfg.get("w12", 0.0)
    return (0.025 * spend
            + m * arr["f1"]
            + ben * arr["BEN"]
            + w5 * arr["f5"]
            - lgd * arr["f11"] * (arr["f1"] + spend / 12)
            - coll * arr["f3"]
            + w12 * arr["f12"])


def top_set(scores):
    return set(np.argpartition(-np.asarray(scores), C.K)[:C.K])


def overlap(a_set, b_set):
    return 100 * len(a_set & b_set) / C.K