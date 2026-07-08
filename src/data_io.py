"""
data_io.py — load the dataset once, build derived arrays, FREEZE the imputer.

Fixes the notebook's biggest reliability bug: the gbm imputer gave 0.67 on one
run and 0.86 on another (thread non-determinism). Here it is trained ONCE and
saved to cache/spend_imp2.parquet; every later run just loads that file, so the
number can never drift between your laptop and Kaggle again.
"""
import numpy as np
import pandas as pd
from . import config as C


def load_data() -> pd.DataFrame:
    cache = C.CACHE / "data.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = pd.read_csv(C.DATA_CSV)
        df.columns = df.columns.astype(str).str.strip()
        df.to_parquet(cache, index=False)
    df["catsum"] = df[C.CATS].sum(axis=1, min_count=5)
    return df


def sink_spend(df: pd.DataFrame) -> np.ndarray:
    """catsum with missing block -> 0 (the 'sink' basis that won calibration)."""
    return df["catsum"].clip(lower=0).fillna(0).values


def imputed_spend(df: pd.DataFrame) -> np.ndarray:
    """
    gbm-imputed spend, trained ONCE and frozen to parquet. Returns the same
    array forever regardless of machine.
    """
    cache = C.CACHE / "spend_imp2.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["spend_imp2"].values

    from sklearn.ensemble import HistGradientBoostingRegressor
    feats = ["f4", "f21", "f13", "f14", "f15", "f16", "f11", "f1", "f12", "f22"]
    okc = df["catsum"].notna()
    Xall = df[feats].rank()
    Xall = Xall.fillna(Xall.median())
    y = df.loc[okc, "catsum"].rank()
    gbm = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xall[okc.values], y)
    pct = pd.Series(gbm.predict(Xall), index=df.index).rank(pct=True).values
    srt = np.sort(df["catsum"].dropna().values)
    imp = srt[np.clip((pct * len(srt)).astype(int), 0, len(srt) - 1)]
    spend = np.where(df["catsum"].notna(), df["catsum"].clip(lower=0).fillna(0), imp)
    pd.DataFrame({"spend_imp2": spend}).to_parquet(cache, index=False)
    return spend
