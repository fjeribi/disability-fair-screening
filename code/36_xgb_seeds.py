"""Seed stability of the gradient-boosted variants (added in the second revision).
Domain-aware XGBoost + EO cut-offs (DAFS) and items-only XGBoost + EO cut-offs (A3) are refitted with ten
fixed seeds (random_state 0-9; subsample = 0.8 makes the fits seed-dependent) and evaluated on NHIS 2025.
The environment (Python, XGBoost, NumPy versions, platform) is recorded with the results.
Output: out/xgb_seeds{SUF}.csv, out/xgb_seeds_summary{SUF}.csv
"""
import os, sys, platform, numpy as np, pandas as pd, xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, "code")
from fairmetrics import evaluate
src = open("code/09_proposed.py").read().split("rows = []")[0]; ns = {"__name__": "lib"}
sys.argv = ["x", "yA", "0.05"]; exec(src, ns)
tr, te, y, w, wn, g_tr, g_te, model, eo = (ns[k] for k in ["tr", "te", "y", "w", "wn", "g_tr", "g_te", "model", "eo_thresholds"])
SUF = ns["SUF"]; rows = []
for seed in range(10):
    for aware, name in [(False, "Items-only XGBoost + EO cut-offs (A3)"), (True, "Domain-aware XGBoost + EO cut-offs (DAFS)")]:
        feats, params = model(aware); params = dict(params, random_state=seed)
        X, Xte = tr[feats].to_numpy(), te[feats].to_numpy(); oof = np.zeros(len(tr))
        for a, b in GroupKFold(5).split(X, y, groups=tr.psu):
            oof[b] = XGBClassifier(**params).fit(X[a], y[a], sample_weight=wn[a]).predict_proba(X[b])[:, 1]
        pte = XGBClassifier(**params).fit(X, y, sample_weight=wn).predict_proba(Xte)[:, 1]
        thr = eo(oof, y, w, g_tr, 0.05); yh = np.array([pte[i] >= thr[g] for i, g in enumerate(g_te)]).astype(int)
        r = evaluate(te, pte, yh, "yA"); r.update(method=name, seed=seed, python=platform.python_version(),
                                                  xgboost=xgboost.__version__, numpy=np.__version__, platform=platform.platform())
        rows.append(r); print(f"seed {seed} {name:45s} BAcc {r['BAcc']:.3f} AOD {r['AOD']:.3f}", flush=True)
d = pd.DataFrame(rows); d.to_csv(f"out/xgb_seeds{SUF}.csv", index=False)
s = d.groupby("method")[["AUC", "BAcc", "EOD", "FPRgap", "AOD"]].agg(["mean", "std", "min", "max", "median"]).round(4)
s.to_csv(f"out/xgb_seeds_summary{SUF}.csv"); print(s.to_string()); print("SEEDS COMPLETE")
