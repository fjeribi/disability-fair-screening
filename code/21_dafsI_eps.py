"""Stage 7d: DAFS-I sensitivity to the fairness tolerance epsilon (no files overwritten)."""
import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "code"); from fairmetrics import evaluate, DOMS
src = open("code/09_proposed.py").read().split("rows = []")[0]; ns = {"__name__": "lib"}
sys.argv = ["x", "yA", "0.05"]; exec(src, ns)
tr, te, y, w, eo = ns["tr"], ns["te"], ns["y"], ns["w"], ns["eo_thresholds"]
ITEMS = [f"phq8{i}" for i in range(1, 9)]
def X(t):
    Z = t[ITEMS].copy()
    for k in DOMS:
        f = t[f"d_{k}"].fillna(0); Z[f"d_{k}"] = f; Z[f"i4x{k}"] = t.phq84 * f; Z[f"i7x{k}"] = t.phq87 * f
    return Z.to_numpy(float)
oof = np.zeros(len(tr))
for a, b in GroupKFold(5).split(tr, y, groups=tr.psu):
    oof[b] = LogisticRegression(C=1.0, max_iter=5000).fit(X(tr.iloc[a]), y[a], sample_weight=w[a]/w[a].mean()).predict_proba(X(tr.iloc[b]))[:, 1]
pte = LogisticRegression(C=1.0, max_iter=5000).fit(X(tr), y, sample_weight=w/w.mean()).predict_proba(X(te))[:, 1]
rows = []
for eps in [0.03, 0.05, 0.10]:
    thr = eo(oof, y, w, tr.fgroup.to_numpy(), eps)
    yh = np.array([pte[i] >= thr[g] for i, g in enumerate(te.fgroup)]).astype(int)
    r = evaluate(te, pte, yh, "yA"); rows.append(dict(epsilon=eps, **{k: round(r[k], 3) for k in ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD"]}))
out = pd.DataFrame(rows); out.to_csv("out/dafsI_epsilon.csv", index=False); print(out.to_string(index=False))
