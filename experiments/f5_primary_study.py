"""experiments/f5_primary_study.py — test f5 as PRIMARY signal (never done on the board)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, top_set, overlap

df = load_data(); arr = prep(df); spend = imputed_spend(df)
R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
champ_v = pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['Prediction'].values
champ = top_set(champ_v)
f1=df.f1.fillna(0).values; f5=df.f5.fillna(0).values; f11=df.f11.fillna(0).values

print("f5-PRIMARY candidates (f5 as MAIN signal, not added to catsum):\n")
cands = {
    "pure f5":            R(f5),
    "f5 + revolve":       0.7*R(f5)+0.3*R(f1),
    "f5 + revolve - risk":0.7*R(f5)+0.3*R(f1)-0.2*R(f11),
    "f5 x 0.5 + catsum x 0.5": 0.5*R(f5)+0.5*R(spend),
    "champion (ref)":     R(champ_v),
}
for n,v in cands.items():
    t=top_set(v); nov=100-overlap(t,champ)
    # who does pure-f5 pick that champion doesn't?
    enter=list(t-champ)
    ge=df.iloc[enter] if enter else df.iloc[:1]
    print(f"  {n:26} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f} f5 {ge.f5.median():>7,.0f}")
print("\nread: pure-f5 will have HIGH novelty (30%+). That's expected - it's a different signal.")
print("We CANNOT know offline if it helps. But it's the one untested structure with real upside.")