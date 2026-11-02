"""Stage 5: domain-aware gradient-boosted variant (DAFS) and ablations A1-A3.
The proposed method, DAFS-I (item-informed logistic score), is implemented in 18_dafs_variants.py
and reuses the cross-fitting and equalized-odds threshold functions defined here.

Components
  (1) domain-aware score: XGBoost on PHQ-8 items + six Washington Group domain flags, survey-weighted,
      monotone in every item (a higher symptom frequency can never lower risk)
  (2) cross-fitted predictions: 5-fold out-of-fold scores grouped by PSU over all training years, so
      thresholds for small domains are estimated on every training respondent rather than a split
  (3) equalized-odds operating points: one threshold per fairness group, chosen jointly to maximise
      survey-weighted balanced accuracy subject to average-odds gap <= epsilon (grid over a shared
      target (TPR*, FPR*) point; each group takes the threshold closest to the target on its ROC)
Ablations: A1 = unaware score + single threshold; A2 = domain-aware score + single threshold;
           A3 = unaware score + EO thresholds; DAFS = domain-aware score + EO thresholds.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, "code")
from fairmetrics import evaluate, DOMS

YCOL = sys.argv[1] if len(sys.argv) > 1 else "yA"
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
import os
SCALE = os.environ.get("SCALE", "dep")
SUF = "" if SCALE == "dep" else "_anx"
d = pd.read_parquet(f"data/proc/ml_{SCALE}.parquet")
ITEMS = [f"phq8{i}" for i in range(1, 9)] if SCALE == "dep" else [f"gad7{i}" for i in range(1, 8)]
FLAGS = [f"d_{k}" for k in DOMS]
tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
y, w = tr[YCOL].to_numpy().astype(int), tr.wt_pooled.to_numpy()
wn = w / w.mean()
g_tr, g_te = tr.fgroup.to_numpy(), te.fgroup.to_numpy()
GROUPS = np.unique(g_tr)


def model(aware):
    feats = ITEMS + (FLAGS if aware else [])
    mono = "(" + ",".join(["1"] * len(ITEMS) + (["0"] * len(FLAGS) if aware else [])) + ")"
    return feats, dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                       monotone_constraints=mono, n_jobs=2, verbosity=0)


def oof_and_test(aware):
    feats, params = model(aware)
    X, Xte = tr[feats].to_numpy(), te[feats].to_numpy()
    oof = np.zeros(len(tr))
    for a, b in GroupKFold(5).split(X, y, groups=tr.psu):
        oof[b] = XGBClassifier(**params).fit(X[a], y[a], sample_weight=wn[a]).predict_proba(X[b])[:, 1]
    full = XGBClassifier(**params).fit(X, y, sample_weight=wn)
    return oof, full.predict_proba(Xte)[:, 1], full


from statutils import roc_pts


def single_thr(s, yy, ww):
    ss, tp, fp = roc_pts(s, yy, ww)
    i = np.argmax(tp - fp)
    return ss[i]


def eo_thresholds(s, yy, ww, g, eps):
    # Same ROC grid and first-maximum rule; cache group weights and vectorise targets.
    Fs = np.linspace(0.01, 0.45, 89)
    thresholds, tprs, fprs, pos, neg = {}, {}, {}, {}, {}
    for gg in GROUPS:
        mask = g == gg
        ss, tp, fp = roc_pts(s[mask], yy[mask], ww[mask])
        indices = np.concatenate([np.argmin(np.abs(tp - T)[None, :] + np.abs(fp[None, :] - Fs[:, None]), axis=1) for T in np.linspace(0.3, 0.95, 66)])
        thresholds[gg], tprs[gg], fprs[gg] = ss[indices], tp[indices], fp[indices]
        pos[gg], neg[gg] = (ww * yy)[mask].sum(), (ww * (1 - yy))[mask].sum()
    aod = np.maximum.reduce([0.5 * (np.abs(tprs[gg] - tprs['none']) + np.abs(fprs[gg] - fprs['none'])) for gg in GROUPS if gg != 'none'])
    tpr = sum(tprs[gg] * pos[gg] for gg in GROUPS) / sum(pos.values())
    fpr = sum(fprs[gg] * neg[gg] for gg in GROUPS) / sum(neg.values())
    objective = np.where(aod <= eps, (tpr + 1 - fpr) / 2, -1)
    i = int(np.argmax(objective))
    return {gg: thresholds[gg][i] for gg in GROUPS} if objective[i] >= 0 else None


rows = []
PRED = {}
cache = {}
for aware in (False, True):
    cache[aware] = oof_and_test(aware)
for name, aware, fair in [("A1 unaware score + single threshold", False, False),
                          ("A2 domain-aware score + single threshold", True, False),
                          ("A3 unaware score + EO thresholds", False, True),
                          ("DAFS (proposed)", True, True)]:
    oof, pte, _ = cache[aware]
    if fair:
        thr = eo_thresholds(oof, y, w, g_tr, EPS)
        yhat = np.array([pte[i] >= thr[g_te[i]] for i in range(len(te))])
    else:
        yhat = pte >= single_thr(oof, y, w)
    r = evaluate(te, pte, yhat.astype(int), YCOL)
    PRED[name] = (pte.astype(float), yhat.astype(int))
    r["method"] = name
    rows.append(r)
    print(f"{name:42s} AUC {r['AUC']:.3f} BAcc {r['BAcc']:.3f} F1 {r['F1']:.3f} | EOD {r['EOD']:.3f} "
          f"FPRgap {r['FPRgap']:.3f} AOD {r['AOD']:.3f} DPD {r['DPD']:.3f}", flush=True)
res = pd.DataFrame(rows)
res.to_csv(f"out/proposed_{YCOL}{SUF}_eps{EPS:.2f}.csv", index=False)
if abs(EPS - 0.05) < 1e-9:
    res.to_csv(f"out/proposed_{YCOL}{SUF}.csv", index=False)
import pickle
pickle.dump(PRED, open(f"out/pred_proposed_{YCOL}{SUF}_eps{EPS:.2f}.pkl", "wb"))
print(res[[c for c in res.columns if c.startswith(("TPR_", "FPR_"))] + ["method"]].round(3).to_string(index=False))
