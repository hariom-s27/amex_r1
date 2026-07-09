"""experiments/batch_churn.py — the missing lever: CHURN/engagement (the untouched category).
Research: CLV = revenue/(1-retention). Calls-to-cancel = churn signal = lower CLV."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from src import config as C
from src.data_io import load_data, sink_spend
from src.scoring import top_set, overlap
import src.submit as sb

df = load_data(); sp = sink_spend(df); ids = df["id"].values
catmiss = df[C.CATS].isna().all(axis=1).values
f1=df.f1.fillna(0).values; f11=df.f11.fillna(0).values
f13=df.f13.fillna(0).values; f14=df.f14.fillna(0).values
f15=df.f15.fillna(0).values; f16=df.f16.fillna(0).values
# candidate engagement/churn columns (PS: calls-to-cancel, logins, #cards, tenure)
f2=df.f2.fillna(0).values; f12=df.f12.fillna(0).values
f20=df.f20.fillna(0).values; f22=df.f22.fillna(0).values; f23=df.f23.fillna(df.f23.median()).values

# CHAMPION BASIS = ben_cab_hi (0.923)
BEN = 35*f13 + f14 + 40*f15 + f16
def base_model(extra):
    v = 0.025*sp + 0.35*f1 - 1.0*BEN - 1.5*f11*(f1 + sp/12) + extra
    return v - 1e9*catmiss

base = base_model(0.0); bt = top_set(base)
champ_v = pd.ExcelFile(C.OUT/"sub17_ben_cab_hi.xlsx").parse("Predictions")["Prediction"].values
assert overlap(bt, top_set(champ_v)) > 99, "basis must match 0.923 champion"

R = lambda x: pd.Series(np.asarray(x,float)).rank(pct=True).values
S = base.std()
# each probe: penalize a churn signal or reward engagement. try each candidate column.
probes = {
    "penalize_f2":   base_model(-0.5*S*R(f2)),    # f2 = calls to cancel? penalize
    "penalize_f2_hi":base_model(-1.5*S*R(f2)),
    "reward_f12":    base_model(+0.3*S*R(f12)),    # f12 = logins? reward engagement
    "penalize_f20":  base_model(-0.5*S*R(f20)),    # f20 = churn signal?
    "reward_tenure": base_model(+0.5*S*R(f23)),    # f23 = tenure, reward loyalty
    "penalize_f22":  base_model(-0.5*S*R(f22)),
}
print("CHURN/ENGAGEMENT PROBES (novelty vs 0.923 champion):")
built=[]
for name,v in probes.items():
    t=top_set(v); nov=100-overlap(t,bt)
    if nov < 0.5:
        print(f"  {name:16} novelty {nov:.1f}%  <-- inert, SKIP"); continue
    ge=df.iloc[list(t-bt)]
    print(f"  {name:16} novelty {nov:4.1f}% | ENTER spend {ge.catsum.median():>7,.0f} f1 {ge.f1.median():>6,.0f}")
    if 1.0 <= nov <= 20:
        sb.write_sub(v, ids, C.OUT/f"sub19_{name}.xlsx", f"s19_{name}")
        sb.framework_guard(C.OUT/f"sub19_{name}.xlsx")
        built.append(name)
print(f"\nbuilt: {built}")
print("These test the ONE attribute category (Engagement) we never used. Upload the")
print("highest-novelty sensible probe first. If churn matters, this is the 2.5pp path.")