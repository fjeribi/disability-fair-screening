"""Stage 4: superiority comparison. Published screening rules and fairness methods re-implemented on
the same data, label and split. Train: NHIS 2019+2022 (75% of PSUs fit, 25% validation for thresholds);
test: NHIS 2025. Features: the eight PHQ-8 items (unless a method needs more). Survey weights used in
fitting and evaluation. Track A label (latent-class, external indicators) is the main target; Track B
(PHQ-8 >= 10) reproduces the questionnaire-labelled setting of Dang et al.

Methods:
  Rules      PHQ-8 >= 10; PHQ-2 >= 3
  Unaware    logistic regression (LR); XGBoost
  Dang 2024  reweighing (RW); disparate-impact remover (DIR); calibrated equalized-odds post-processing
             (CPP, Pleiss et al. 2017, FNR cost, multigroup); population sensitivity-guided threshold
             adjustment (PSTA, group thresholds matching overall sensitivity)
  Shipman    fairness-constrained XGBoost via Lagrangian (dual) reduction, TPR parity
  Tian       fairness-constrained stacking ensemble (LR, RF, XGB -> EO-constrained meta-learner)
  Khan       fairness-regularised model (loss + lambda * equalized-odds penalty), tabular adaptation
"""
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity, EqualizedOdds

sys.path.insert(0, "code")
from fairmetrics import evaluate

YCOL = sys.argv[1] if len(sys.argv) > 1 else "yA"
rng = np.random.default_rng(11)
import os
SCALE = os.environ.get("SCALE", "dep")          # dep = PHQ-8 (main), anx = GAD-7 extension
SUF = "" if SCALE == "dep" else "_anx"
d = pd.read_parquet(f"data/proc/ml_{SCALE}.parquet")
ITEMS = [f"phq8{i}" for i in range(1, 9)] if SCALE == "dep" else [f"gad7{i}" for i in range(1, 8)]
tr_all, te = d[d.split == "train"].copy(), d[d.split == "test"].copy()
psus = tr_all.psu.unique()
val_psu = set(rng.choice(psus, int(0.25 * len(psus)), replace=False))
va = tr_all[tr_all.psu.isin(val_psu)].copy()
tr = tr_all[~tr_all.psu.isin(val_psu)].copy()
Xtr, Xva, Xte = tr[ITEMS].to_numpy(), va[ITEMS].to_numpy(), te[ITEMS].to_numpy()
ytr, yva = tr[YCOL].to_numpy().astype(int), va[YCOL].to_numpy().astype(int)
wtr, wva = tr.wt_pooled.to_numpy(), va.wt_pooled.to_numpy()
wtr_n = wtr / wtr.mean()
gtr, gva, gte = tr.fgroup.to_numpy(), va.fgroup.to_numpy(), te.fgroup.to_numpy()


def youden_thr(y, s, w):
    best, bt = -1, 0.5
    for t in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197))):
        p = s >= t
        tpr = w[(y == 1) & p].sum() / w[y == 1].sum()
        fpr = w[(y == 0) & p].sum() / w[y == 0].sum()
        if tpr - fpr > best:
            best, bt = tpr - fpr, t
    return bt


results = []
PRED = {}


def record(name, s_te, yhat_te):
    PRED[name] = (np.asarray(s_te, float), np.asarray(yhat_te, float))
    r = evaluate(te, s_te, yhat_te.astype(float), YCOL)
    r["method"] = name
    results.append(r)
    print(f"{name:34s} AUC {r['AUC']:.3f} BAcc {r['BAcc']:.3f} F1 {r['F1']:.3f} | EOD {r['EOD']:.3f} "
          f"FPRgap {r['FPRgap']:.3f} AOD {r['AOD']:.3f} DPD {r['DPD']:.3f}", flush=True)


# ---- screening rules ----
s = te[ITEMS].sum(axis=1).to_numpy()
record("PHQ-8 >= 10 (standard)" if SCALE == "dep" else "GAD-7 >= 10 (standard)", s, s >= 10)
s2 = te[ITEMS[:2]].sum(axis=1).to_numpy()
record("PHQ-2 >= 3" if SCALE == "dep" else "GAD-2 >= 3", s2, s2 >= 3)

# ---- unaware models ----
lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr, sample_weight=wtr_n)
t = youden_thr(yva, lr.predict_proba(Xva)[:, 1], wva)
p_lr_te = lr.predict_proba(Xte)[:, 1]
record("Logistic regression (unaware)", p_lr_te, p_lr_te >= t)
xgb_params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, n_jobs=2, verbosity=0, random_state=0)
xg = XGBClassifier(**xgb_params).fit(Xtr, ytr, sample_weight=wtr_n)
t = youden_thr(yva, xg.predict_proba(Xva)[:, 1], wva)
p = xg.predict_proba(Xte)[:, 1]
record("XGBoost (unaware)", p, p >= t)

# ---- Dang 2024: RW ----
w_rw = wtr_n.copy()
for g in np.unique(gtr):
    for yy in (0, 1):
        m = (gtr == g) & (ytr == yy)
        exp = wtr_n[gtr == g].sum() * wtr_n[ytr == yy].sum() / wtr_n.sum()
        if wtr_n[m].sum() > 0:
            w_rw[m] *= exp / wtr_n[m].sum()
m_rw = LogisticRegression(max_iter=2000).fit(Xtr, ytr, sample_weight=w_rw)
t = youden_thr(yva, m_rw.predict_proba(Xva)[:, 1], wva)
p = m_rw.predict_proba(Xte)[:, 1]
record("Dang: Reweighing (RW)", p, p >= t)


# ---- Dang 2024: DIR (repair level 1, quantile mapping to pooled distribution) ----
def dir_repair(X, g, ref_X):
    # All group CDFs and the target distribution are fitted on training data only.
    Xr = X.astype(float).copy()
    for j in range(X.shape[1]):
        order=np.argsort(ref_X[:,j],kind="stable"); pooled=ref_X[order,j]
        for gg in np.unique(g):
            train=ref_X[gtr==gg,j]; m=g==gg
            vals,counts=np.unique(train,return_counts=True)
            # Mid-CDF, interpolated from the fixed training support.
            q=(np.cumsum(counts)-counts/2)/counts.sum()
            ranks=np.interp(X[m,j],vals,q,left=0,right=1)
            Xr[m,j]=np.quantile(pooled,ranks)
    return Xr


m_dir = LogisticRegression(max_iter=2000).fit(dir_repair(Xtr, gtr, Xtr), ytr, sample_weight=wtr_n)
t = youden_thr(yva, m_dir.predict_proba(dir_repair(Xva, gva, Xtr))[:, 1], wva)
p = m_dir.predict_proba(dir_repair(Xte, gte, Xtr))[:, 1]
record("Dang: Disparate impact remover", p, p >= t)


# ---- Dang 2024: CPP (calibrated equalized odds, FNR cost, multigroup) ----
def cpp_fit(s, y, g, w):
    gen_fnr = {gg: np.average(1 - s[(g == gg) & (y == 1)], weights=w[(g == gg) & (y == 1)]) for gg in np.unique(g)}
    base = {gg: np.average(y[g == gg], weights=w[g == gg]) for gg in np.unique(g)}
    target = max(gen_fnr.values())
    alpha = {}
    for gg in gen_fnr:
        trivial_fnr = 1 - base[gg]
        alpha[gg] = 0.0 if abs(trivial_fnr - gen_fnr[gg]) < 1e-9 else \
            float(np.clip((target - gen_fnr[gg]) / (trivial_fnr - gen_fnr[gg]), 0, 1))
    return alpha, base


def cpp_apply(s, g, alpha, base, rng):
    out = s.copy()
    for gg in alpha:
        m = g == gg
        flip = rng.random(m.sum()) < alpha[gg]
        out[np.where(m)[0][flip]] = base[gg]
    return out


p_lr_va = lr.predict_proba(Xva)[:, 1]
alpha, base = cpp_fit(p_lr_va, yva, gva, wva)
s_va = cpp_apply(p_lr_va, gva, alpha, base, np.random.default_rng(1))
t = youden_thr(yva, s_va, wva)
s_te = cpp_apply(p_lr_te, gte, alpha, base, np.random.default_rng(2))
# Evaluate the intended randomized mixing rule in expectation rather than one draw.
q=np.array([(1-alpha[g])*(p_lr_te[i]>=t)+alpha[g]*(base[g]>=t) for i,g in enumerate(gte)])
s_mean=np.array([(1-alpha[g])*p_lr_te[i]+alpha[g]*base[g] for i,g in enumerate(gte)])
record("Dang: Calibrated EO post-proc (CPP)", s_mean, q)

# ---- Dang 2024: PSTA (group thresholds so each group's sensitivity matches overall) ----
t_all = youden_thr(yva, p_lr_va, wva)
target_tpr = wva[(yva == 1) & (p_lr_va >= t_all)].sum() / wva[yva == 1].sum()
thr = {}
for gg in np.unique(gva):
    m = (gva == gg) & (yva == 1)
    ss, ww = p_lr_va[m], wva[m]
    o = np.argsort(-ss)
    cum = np.cumsum(ww[o]) / ww.sum()
    thr[gg] = ss[o][min(np.searchsorted(cum, target_tpr), len(o) - 1)]
yhat = np.array([p_lr_te[i] >= thr.get(gte[i], t_all) for i in range(len(gte))])
record("Dang: PSTA group thresholds", p_lr_te, yhat)

# ---- Shipman 2026: fairness-constrained XGBoost (Lagrangian / dual reduction, TPR parity) ----
eg = ExponentiatedGradient(XGBClassifier(**xgb_params), constraints=TruePositiveRateParity(difference_bound=0.02))
# fairlearn reductions take no sample weights: fit on a survey-weight-proportional resample
ridx = np.random.default_rng(5).choice(len(Xtr), len(Xtr), replace=True, p=wtr / wtr.sum())
eg.fit(Xtr[ridx], ytr[ridx], sensitive_features=gtr[ridx])
t = youden_thr(yva, eg._pmf_predict(Xva)[:, 1], wva)
p = eg._pmf_predict(Xte)[:, 1]
record("Shipman: fair XGBoost (dual, TPR parity)", p, p)

# ---- Tian 2026: fairness-constrained stacking ensemble ----
base_models = [LogisticRegression(max_iter=2000),
               RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, n_jobs=2, random_state=0),
               XGBClassifier(**xgb_params)]
for bm in base_models:
    bm.fit(Xtr, ytr, sample_weight=wtr_n)
Zva = np.column_stack([bm.predict_proba(Xva)[:, 1] for bm in base_models])
Zte = np.column_stack([bm.predict_proba(Xte)[:, 1] for bm in base_models])
meta = ExponentiatedGradient(LogisticRegression(max_iter=2000), constraints=EqualizedOdds(difference_bound=0.02))
vidx = np.random.default_rng(6).choice(len(Zva), len(Zva), replace=True, p=wva / wva.sum())
# Fit the meta-learner on PSU-cross-fitted base predictions within the training partition.
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
Zmeta=np.zeros((len(tr),len(base_models)))
for ia,ib in GroupKFold(3).split(Xtr,ytr,groups=tr.psu):
    for jj,bm in enumerate(base_models):
        fm=clone(bm).fit(Xtr[ia],ytr[ia],sample_weight=wtr_n[ia]);Zmeta[ib,jj]=fm.predict_proba(Xtr[ib])[:,1]
vidx=np.random.default_rng(6).choice(len(tr),len(tr),replace=True,p=wtr/wtr.sum())
meta.fit(Zmeta[vidx], ytr[vidx], sensitive_features=gtr[vidx])
t = youden_thr(yva, meta._pmf_predict(Zva)[:, 1], wva)
p = meta._pmf_predict(Zte)[:, 1]
record("Tian: fair stacking ensemble (FCSE)", p, p)

# ---- Khan 2026: fairness-regularised model (tabular adaptation) ----
Xs = np.column_stack([np.ones(len(Xtr)), Xtr])
G = pd.get_dummies(gtr).to_numpy(float)


def khan_loss(b, lam=5.0):
    z = Xs @ b
    pr = expit(z)
    ll = -np.sum(wtr_n * (ytr * np.log(pr + 1e-12) + (1 - ytr) * np.log(1 - pr + 1e-12))) / wtr_n.sum()
    pen = 0.0
    for yy in (0, 1):
        m = ytr == yy
        means = (G[m] * (wtr_n[m] * pr[m])[:, None]).sum(0) / ((G[m] * wtr_n[m][:, None]).sum(0) + 1e-12)
        overall = np.average(pr[m], weights=wtr_n[m])
        pen += np.sum((means - overall) ** 2)
    return ll + lam * pen


b = minimize(khan_loss, np.zeros(Xs.shape[1]), method="L-BFGS-B").x
p_va = expit(np.column_stack([np.ones(len(Xva)), Xva]) @ b)
t = youden_thr(yva, p_va, wva)
p = expit(np.column_stack([np.ones(len(Xte)), Xte]) @ b)
record("Khan: fairness-regularised model", p, p >= t)
if not b.shape: raise RuntimeError("Invalid regularised fit")

pd.DataFrame(results).to_csv(f"out/baselines_{YCOL}{SUF}.csv", index=False)
import pickle
pickle.dump(PRED, open(f"out/pred_baselines_{YCOL}{SUF}.pkl", "wb"))
