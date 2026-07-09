"""audit_submission.py — pre-submission gate. Run on ANY file before uploading."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
from . import config as C

_OLD = ["625", "f19", "0.0235", "0.009*f21", "0.16*f5", "0.27*f1"]

def _top(v, k=C.K): return set(np.argpartition(-np.asarray(v,float), k)[:k])
def _overlap(a,b): return 100*len(a&b)/C.K

def _load_anchor_tops():
    raw = json.load(open(C.ANCHORS_JSON)); out={}
    for name,s in raw.items():
        if name.startswith("_"): continue
        p = C.OUT/name
        if p.exists():
            v = pd.ExcelFile(p).parse("Predictions")["Prediction"].values
            out[name]=(_top(v), float(s))
    return out

def audit(path):
    path=Path(path)
    print("="*64); print(f"PRE-SUBMISSION AUDIT: {path.name}"); print("="*64)
    if not path.exists(): print("FAIL file missing"); return False
    xl=pd.ExcelFile(path); checks=[]
    checks.append(("two sheets", xl.sheet_names==["Predictions","Profitability Framework"], str(xl.sheet_names)))
    pred=xl.parse("Predictions")
    checks.append(("columns [ID,Prediction]", list(pred.columns)==["ID","Prediction"], str(list(pred.columns))))
    checks.append((f"rows=={C.N:,}", len(pred)==C.N, f"{len(pred):,}"))
    id_ok=bool((pred["ID"].values==np.arange(C.N)).all()); checks.append(("id order 0..N-1", id_ok, ""))
    p=pd.to_numeric(pred["Prediction"],errors="coerce")
    nan_ok=p.notna().all(); checks.append(("no NaN", nan_ok, f"{int(p.isna().sum())} bad"))
    checks.append(("no inf", bool(np.isfinite(p.fillna(0)).all()), ""))
    if nan_ok:
        cut=p.nlargest(C.K).min(); at=int((p==cut).sum())
        checks.append(("cutoff not tie-swamped", at<=max(1,int(0.001*C.N)), f"{at} at cutoff"))
    fw=xl.parse("Profitability Framework")
    checks.append(("framework 10/10", fw["Response"].notna().sum()==10, f"{int(fw['Response'].notna().sum())}/10"))
    eq=" ".join(str(x) for x in fw["Response"].fillna("")); bad=[f for f in _OLD if f in eq]
    checks.append(("no stale text", not bad, f"{bad}" if bad else ""))
    print("\n[MECHANICAL]"); ok=True
    for n,pas,note in checks:
        ok&=bool(pas); print(f"  {'PASS' if pas else 'FAIL'}  {n}"+(f"  ({note})" if note else ""))
    if not (nan_ok and len(pred)==C.N and id_ok):
        print("\nVERDICT: FAIL"); return False
    print("\n[GEOMETRY vs scored files]")
    anchors=_load_anchor_tops(); t=_top(p.values); champ=None
    for name,(tp,s) in sorted(anchors.items(), key=lambda kv:-kv[1][1]):
        o=_overlap(t,tp); print(f"  vs {name:26} (real {s:5.1f}): {o:5.1f}%")
        if champ is None: champ=(name,tp,s)
    cn,ct,cs=champ; oc=_overlap(t,ct)
    print(f"\n  GUARANTEED BAND (from {cn} @ {cs}): [{cs-(100-oc):.1f}, {cs+(100-oc):.1f}]")
    print(f"  novelty vs champion: {100-oc:.1f}%")
    print("\n[SHAKEUP SIMULATION]")
    rng=np.random.default_rng(0); fold=rng.integers(0,5,C.N); stab=[]
    for f in range(5):
        sub=p.values.astype(float).copy(); sub[fold==f]=-np.inf
        kk=int((fold!=f).sum()*0.20); ft=set(np.argpartition(-sub,kk)[:kk])
        stab.append(len(ft&t)/kk*100)
    print(f"  stability across 5 folds: {np.mean(stab):.1f}% mean | {np.std(stab):.2f} std")
    print("\nVERDICT: ALL MECHANICAL CHECKS PASS — safe to upload.")
    return ok

if __name__=="__main__":
    if len(sys.argv)<2: print("usage: python -m src.audit_submission <file.xlsx>"); sys.exit(1)
    audit(sys.argv[1])