"""config.py — the ONLY place paths and constants live. Edit paths here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "outputs" / "submissions"
LOG = ROOT / "outputs" / "experiment_log.csv"
ANCHORS_JSON = Path(__file__).resolve().parent / "anchors.json"

DATA_CSV = RAW / "r1_datset.csv"
TEMPLATE = RAW / "r1_submission_template.xlsx"

N = 500_000          # rows
K = 100_000          # top-20%
CATS = ["f6", "f7", "f8", "f9", "f10"]
FEATURES = [f"f{i}" for i in range(1, 24)]

for d in (CACHE, OUT):
    d.mkdir(parents=True, exist_ok=True)
