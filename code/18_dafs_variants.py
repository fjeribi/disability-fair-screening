"""Stage 7c: DAFS-I, the proposed method, and its paired comparison with the ablations.
DAFS-I: logistic score on items + domain flags + (biased item) x domain interactions, cross-fitted,
survey-weighted, same EO thresholds. Biased items follow the pre-specified rule (ratio of odds ratios < 1,
p < 0.05 in >= 3 domains): PHQ-8 items 4 and 7 (depression, SCALE=dep); GAD-7 item 1 (anxiety, SCALE=anx).
Paired bootstrap vs A3 and vs DAFS (XGBoost) on NHIS 2025."""
import sys, pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "code"); from fairmetrics import evaluate, DOMS
src = open("code/09_proposed.py").read().split("rows = []")[0]; ns = {"__name__": "lib"}
sys.argv = ["x", "yA", "0.05"]; exec(src, ns)
tr, te, y, w, eo = ns["tr"], ns["te"], ns["y"], ns["w"], ns["eo_thresholds"]
SCALE, SUF, ITEMS = ns["SCALE"], ns["SUF"], ns["ITEMS"]
BIASED = ["phq84", "phq87"] if SCALE == "dep" else pd.read_csv("out/gad_selected_items.csv")["item"].tolist()
def X(t):
    Z = t[ITEMS].copy()
    for k in DOMS:
        f = t[f"d_{k}"].fillna(0); Z[f"d_{k}"] = f
        for it in BIASED:
            Z[f"{it}x{k}"] = t[it] * f
    return Z.to_numpy(float)
oof = np.zeros(len(tr))
for a, b in GroupKFold(5).split(tr, y, groups=tr.psu):
    oof[b] = LogisticRegression(C=1.0, max_iter=5000).fit(X(tr.iloc[a]), y[a], sample_weight=w[a]/w[a].mean()).predict_proba(X(tr.iloc[b]))[:, 1]
m = LogisticRegression(C=1.0, max_iter=5000).fit(X(tr), y, sample_weight=w/w.mean())
pte = m.predict_proba(X(te))[:, 1]
thr = eo(oof, y, w, tr.fgroup.to_numpy(), 0.05)
yh = np.array([pte[i] >= thr[g] for i, g in enumerate(te.fgroup)]).astype(int)
r = evaluate(te, pte, yh, "yA")
print(f"DAFS-I  AUC {r['AUC']:.3f} BAcc {r['BAcc']:.3f} F1 {r['F1']:.3f} | EOD {r['EOD']:.3f} FPRgap {r['FPRgap']:.3f} AOD {r['AOD']:.3f}")
pp = pickle.load(open(f"out/pred_proposed_yA{SUF}_eps0.05.pkl", "rb")); pp["DAFS-I"] = (pte, yh)
pickle.dump({"DAFS-I": (pte, yh)}, open(f"out/pred_dafsI{SUF}.pkl", "wb"))
pickle.dump({"oof": oof, "pte": pte, "thr": thr}, open(f"out/dafsI_scores{SUF}.pkl", "wb"))
pd.DataFrame({"group": list(thr), "threshold": [float(v) for v in thr.values()]}).to_csv(f"out/dafsI_thresholds{SUF}.csv", index=False)
rng = np.random.default_rng(9); psus = te.groupby("stratum")["psu"].unique(); ix = te.groupby("psu").indices
D = {k: [] for k in ["DAFS-I", "DAFS (proposed)", "A3 unaware score + EO thresholds"]}
for b in range(300):
    ii = np.concatenate([ix[p] for p in np.concatenate([rng.choice(v, len(v)) for v in psus.values])])
    tb = te.iloc[ii].reset_index(drop=True)
    for k in D: s, h = pp[k]; D[k].append(evaluate(tb, s[ii], h[ii], "yA"))
for a, b in [("DAFS-I", "A3 unaware score + EO thresholds"), ("DAFS (proposed)", "A3 unaware score + EO thresholds"), ("DAFS-I", "DAFS (proposed)")]:
    out = []
    for k in ["BAcc", "EOD", "FPRgap", "AOD"]:
        diff = np.array([x[k] for x in D[a]]) - np.array([x[k] for x in D[b]])
        p = (diff >= 0).mean() if k != "BAcc" else (diff <= 0).mean()
        out.append(f"{k} {np.median(diff):+.3f} (p={p:.3f})")
    print(f"{a} vs {b}: " + "; ".join(out))
