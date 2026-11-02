"""Full-resource comparator arm (added in the second revision).

Every published comparator receives the same training resources as the cross-fitted proposed models:
all development respondents are used for fitting, and every quantity that the original procedure
estimates on held-out data (Youden threshold, CPP mixing rates, PSTA group thresholds, stacking
meta-learner) is estimated on 5-fold PSU-grouped out-of-fold predictions, using exactly the same
GroupKFold(5) folds as 09_proposed.py. Final models are refitted on all development data and applied to
the 2025 test year. Randomised reductions (Shipman, Tian) are evaluated through their intended decision
probabilities, as in the matched arm.
Output: out/fullres_baselines_yA{SUF}.csv and out/pred_fullres_baselines_yA{SUF}.pkl
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity, EqualizedOdds

sys.path.insert(0, "code")
from fairmetrics import evaluate

YCOL = "yA"
SCALE = os.environ.get("SCALE", "dep"); SUF = "" if SCALE == "dep" else "_anx"
d = pd.read_parquet(f"data/proc/ml_{SCALE}.parquet")
ITEMS = [f"phq8{i}" for i in range(1, 9)] if SCALE == "dep" else [f"gad7{i}" for i in range(1, 8)]
tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
X, Xte = tr[ITEMS].to_numpy(float), te[ITEMS].to_numpy(float)
y, w = tr[YCOL].to_numpy().astype(int), tr.wt_pooled.to_numpy()
wn = w / w.mean()
g, gte = tr.fgroup.to_numpy(), te.fgroup.to_numpy()
FOLDS = list(GroupKFold(5).split(X, y, groups=tr.psu))          # identical to 09_proposed.py
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, n_jobs=2, verbosity=0, random_state=0)
results, PRED = [], {}


def youden_thr(yy, s, ww):
    best, bt = -1, 0.5
    for t in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197))):
        p = s >= t
        v = ww[(yy == 1) & p].sum() / ww[yy == 1].sum() - ww[(yy == 0) & p].sum() / ww[yy == 0].sum()
        if v > best:
            best, bt = v, t
    return bt


def record(name, s_te, q_te):
    PRED[name] = (np.asarray(s_te, float), np.asarray(q_te, float))
    r = evaluate(te, s_te, np.asarray(q_te, float), YCOL); r["method"] = name; results.append(r)
    print(f"{name:42s} AUC {r['AUC']:.4f} BAcc {r['BAcc']:.4f} AOD {r['AOD']:.4f}", flush=True)


def oof_and_full(fit, predict, Xtr=X, Xtest=Xte):
    """fit(Xa, ya, wa, ga) -> model; predict(model, X) -> scores. OOF over FOLDS, then refit on all."""
    oof = np.zeros(len(Xtr))
    for a, b in FOLDS:
        oof[b] = predict(fit(Xtr[a], y[a], wn[a], g[a]), Xtr[b])
    full = fit(Xtr, y, wn, g)
    return oof, predict(full, Xtest), full


proba = lambda m, Z: m.predict_proba(Z)[:, 1]
lr_fit = lambda Xa, ya, wa, ga: LogisticRegression(max_iter=2000).fit(Xa, ya, sample_weight=wa)

# ---- unaware models ----
oof_lr, te_lr, _ = oof_and_full(lr_fit, proba)
t = youden_thr(y, oof_lr, w); record("Logistic regression (unaware)", te_lr, te_lr >= t)
oof_x, te_x, _ = oof_and_full(lambda Xa, ya, wa, ga: XGBClassifier(**XGB).fit(Xa, ya, sample_weight=wa), proba)
t = youden_thr(y, oof_x, w); record("XGBoost (unaware)", te_x, te_x >= t)


# ---- Dang: reweighing ----
def rw_fit(Xa, ya, wa, ga):
    wr = wa.copy()
    for gg in np.unique(ga):
        for yy in (0, 1):
            m = (ga == gg) & (ya == yy)
            if wa[m].sum() > 0:
                wr[m] *= (wa[ga == gg].sum() * wa[ya == yy].sum() / wa.sum()) / wa[m].sum()
    return LogisticRegression(max_iter=2000).fit(Xa, ya, sample_weight=wr)


oof, s, _ = oof_and_full(rw_fit, proba)
t = youden_thr(y, oof, w); record("Dang: Reweighing (RW)", s, s >= t)


# ---- Dang: disparate-impact remover; group CDFs and target distribution from the fitting data only ----
def dir_transform(Xq, gq, Xref, gref):
    Xr = Xq.astype(float).copy()
    for j in range(Xq.shape[1]):
        pooled = np.sort(Xref[:, j])
        for gg in np.unique(gq):
            ref = Xref[gref == gg, j] if (gref == gg).any() else Xref[:, j]
            vals, counts = np.unique(ref, return_counts=True)
            q = (np.cumsum(counts) - counts / 2) / counts.sum()
            m = gq == gg
            Xr[m, j] = np.quantile(pooled, np.interp(Xq[m, j], vals, q, left=0, right=1))
    return Xr


class DIR:
    def __init__(self, Xa, ya, wa, ga):
        self.Xa, self.ga = Xa, ga
        self.m = LogisticRegression(max_iter=2000).fit(dir_transform(Xa, ga, Xa, ga), ya, sample_weight=wa)


oof = np.zeros(len(X))
for a, b in FOLDS:
    mdl = DIR(X[a], y[a], wn[a], g[a]); oof[b] = proba(mdl.m, dir_transform(X[b], g[b], mdl.Xa, mdl.ga))
mdl = DIR(X, y, wn, g); s = proba(mdl.m, dir_transform(Xte, gte, X, g))
t = youden_thr(y, oof, w); record("Dang: Disparate impact remover", s, s >= t)

# ---- Dang: calibrated equalized-odds post-processing (parameters from OOF LR scores) ----
gen_fnr = {gg: np.average(1 - oof_lr[(g == gg) & (y == 1)], weights=w[(g == gg) & (y == 1)]) for gg in np.unique(g)}
base = {gg: np.average(y[g == gg], weights=w[g == gg]) for gg in np.unique(g)}
target = max(gen_fnr.values())
alpha = {gg: 0.0 if abs((1 - base[gg]) - gen_fnr[gg]) < 1e-9 else
         float(np.clip((target - gen_fnr[gg]) / ((1 - base[gg]) - gen_fnr[gg]), 0, 1)) for gg in gen_fnr}
mix_oof = np.array([(1 - alpha[gg]) * oof_lr[i] + alpha[gg] * base[gg] for i, gg in enumerate(g)])
t = youden_thr(y, mix_oof, w)
q = np.array([(1 - alpha[gg]) * (te_lr[i] >= t) + alpha[gg] * (base[gg] >= t) for i, gg in enumerate(gte)])
s_mean = np.array([(1 - alpha[gg]) * te_lr[i] + alpha[gg] * base[gg] for i, gg in enumerate(gte)])
record("Dang: Calibrated EO post-proc (CPP)", s_mean, q)

# ---- Dang: PSTA (group thresholds matching overall sensitivity, from OOF LR scores) ----
t_all = youden_thr(y, oof_lr, w)
target_tpr = w[(y == 1) & (oof_lr >= t_all)].sum() / w[y == 1].sum()
thr = {}
for gg in np.unique(g):
    m = (g == gg) & (y == 1); ss, ww = oof_lr[m], w[m]; o = np.argsort(-ss); cum = np.cumsum(ww[o]) / ww.sum()
    thr[gg] = ss[o][min(np.searchsorted(cum, target_tpr), len(o) - 1)]
record("Dang: PSTA group thresholds", te_lr, np.array([te_lr[i] >= thr.get(gg, t_all) for i, gg in enumerate(gte)]))

# ---- Shipman: fairness-constrained XGBoost (reduction; intended decision probabilities) ----
ridx = np.random.default_rng(5).choice(len(X), len(X), replace=True, p=w / w.sum())
eg = ExponentiatedGradient(XGBClassifier(**XGB), constraints=TruePositiveRateParity(difference_bound=0.02))
eg.fit(X[ridx], y[ridx], sensitive_features=g[ridx])
p = eg._pmf_predict(Xte)[:, 1]; record("Shipman: fair XGBoost (dual, TPR parity)", p, p)

# ---- Tian: fair stacking; meta-learner fitted on OOF base predictions over all development data ----
bases = [LogisticRegression(max_iter=2000),
         RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, n_jobs=2, random_state=0),
         XGBClassifier(**XGB)]
Z = np.zeros((len(X), len(bases)))
for a, b in FOLDS:
    for j, bm in enumerate(bases):
        Z[b, j] = proba(clone(bm).fit(X[a], y[a], sample_weight=wn[a]), X[b])
Zte = np.column_stack([proba(clone(bm).fit(X, y, sample_weight=wn), Xte) for bm in bases])
meta = ExponentiatedGradient(LogisticRegression(max_iter=2000), constraints=EqualizedOdds(difference_bound=0.02))
vidx = np.random.default_rng(6).choice(len(X), len(X), replace=True, p=w / w.sum())
meta.fit(Z[vidx], y[vidx], sensitive_features=g[vidx])
p = meta._pmf_predict(Zte)[:, 1]; record("Tian: fair stacking ensemble (FCSE)", p, p)

# ---- Khan: fairness-regularised model (threshold from OOF scores) ----
def khan_fit(Xa, ya, wa, ga, lam=5.0):
    Xs = np.column_stack([np.ones(len(Xa)), Xa]); G = pd.get_dummies(ga).to_numpy(float)

    def loss(b):
        pr = expit(Xs @ b)
        ll = -np.sum(wa * (ya * np.log(pr + 1e-12) + (1 - ya) * np.log(1 - pr + 1e-12))) / wa.sum()
        pen = 0.0
        for yy in (0, 1):
            m = ya == yy
            means = (G[m] * (wa[m] * pr[m])[:, None]).sum(0) / ((G[m] * wa[m][:, None]).sum(0) + 1e-12)
            pen += np.sum((means - np.average(pr[m], weights=wa[m])) ** 2)
        return ll + lam * pen
    return minimize(loss, np.zeros(Xs.shape[1]), method="L-BFGS-B").x


kpred = lambda b, Z: expit(np.column_stack([np.ones(len(Z)), Z]) @ b)
oof, s, _ = oof_and_full(khan_fit, kpred)
t = youden_thr(y, oof, w); record("Khan: fairness-regularised model", s, s >= t)

pd.DataFrame(results).to_csv(f"out/fullres_baselines_{YCOL}{SUF}.csv", index=False)
pickle.dump(PRED, open(f"out/pred_fullres_baselines_{YCOL}{SUF}.pkl", "wb"))
print("FULL-RESOURCE COMPLETE", SCALE)
