"""experiments/risk_study.py — test the research-backed 'stronger risk penalty' hypothesis."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, imputed_spend
from src.scoring import prep, build, top_set, overlap

df = load_data(); arr = prep(df); spend = imputed_spend(df)
R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
champ_v = pd.ExcelFile(C.OUT/'sub04_impute2.xlsx').parse('Predictions')['Prediction'].values
champ = top_set(champ_v)
f1=df.f1.fillna(0).values; f11=df.f11.fillna(0).values

print("=== industry model: profit = interchange + interest(revolvers) - rewards - LOSSES ===")
print("champion has weak risk. Test stronger risk penalties + revolver-risk interaction.\n")

# candidates with escalating risk weight (all else = champion shape)
def model(lgd, interact=False):
    risk = f11*(f1 + spend/12)
    if interact:                      # research: revolvers riskier -> extra penalty when revolving
        risk = risk * (1 + (f1>0))
    return 0.025*spend + 0.22*f1 - lgd*risk

print("model                    novelty-vs-champ | seg4(distress) picked%")
seg4_mask = (f11 > 0.03) & (f1 > 0)   # the high-risk revolver segment
for lgd in [0.85, 1.5, 3.0, 6.0]:
    t = top_set(model(lgd))
    nov = 100 - overlap(t, champ)
    inT = np.zeros(C.N,bool); inT[list(t)]=True
    print(f"  lgd={lgd:4}                  {nov:4.1f}%           {100*inT[seg4_mask].mean():4.1f}%")
for lgd in [1.5, 3.0]:
    t = top_set(model(lgd, interact=True))
    nov = 100 - overlap(t, champ)
    inT = np.zeros(C.N,bool); inT[list(t)]=True
    print(f"  lgd={lgd} + revolver-risk    {nov:4.1f}%           {100*inT[seg4_mask].mean():4.1f}%")

print("\nchampion seg4 picked%:", end=" ")
inC=np.zeros(C.N,bool); inC[list(champ)]=True
print(f"{100*inC[seg4_mask].mean():.1f}%  (we want to push this DOWN if risk matters)")
print("\nread: pick the lgd with 8-20% novelty that cleanly drops seg4. That's submission #5 candidate.")