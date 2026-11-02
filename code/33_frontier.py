"""Fairness-accuracy frontier analysis (added in the second revision).

Question: at the same accuracy, which method gives the smaller fairness gap?
Every method is turned into a family of decision rules on development data only, and one common selection
rule is applied to every family: for each accuracy level b, take the rule with the smallest development
average odds difference (AOD) among rules whose development balanced accuracy (BAcc) is at least b.
Families (all from 5-fold PSU-grouped out-of-fold scores on the same folds as DAFS-I):
  proposed  : group cut-offs from the equalized-odds grid (66 TPR targets x 89 FPR targets) on the
              DAFS-I score (PHQ-8) or the items-only score (GAD-7 primary; DAFS-I also reported)
  PSTA      : group thresholds giving every group the same sensitivity s, s on a grid (Dang et al., 2024)
  RW, DIR, Khan, LR : single threshold on a quantile grid of the method's score
The selected rules are then evaluated once on NHIS 2025 across the overlapping domains, with 500 Rao-Wu
replicates shared by all methods (seed 842, as in the main analysis). Reported: test BAcc and AOD per level,
paired differences proposed minus comparator per level, and their mean over the common levels.
Narrow-band variant (BAND=0.005): rules whose development BAcc lies within +/-0.0025 of each level.
Outputs: out/frontier_points_{scale}.csv, out/frontier_diffs_{scale}.csv, out/frontier_rules_{scale}.pkl
"""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, "code")
from statutils import roc_pts, bootstrap_weights
from fairmetrics import DOMS

SCALE = os.environ.get("SCALE", "dep"); SUF = "" if SCALE == "dep" else "_anx"
BAND = float(os.environ.get("BAND", 0))   # > 0: narrow-band variant, rules with |dev BAcc - level| <= BAND/2
TAG = SCALE + ("_band" if BAND else "")
d = pd.read_parquet(f"data/proc/ml_{SCALE}.parquet")
ITEMS = [f"phq8{i}" for i in range(1, 9)] if SCALE == "dep" else [f"gad7{i}" for i in range(1, 8)]
tr, te = d[d.split == "train"].reset_index(drop=True), d[d.split == "test"].reset_index(drop=True)
X, Xte = tr[ITEMS].to_numpy(float), te[ITEMS].to_numpy(float)
y, w = tr.yA.to_numpy().astype(int), tr.wt_pooled.to_numpy(); wn = w / w.mean()
g, gte = tr.fgroup.to_numpy(), te.fgroup.to_numpy(); GROUPS = np.unique(g)
FOLDS = list(GroupKFold(5).split(X, y, groups=tr.psu))
LEVELS = np.round(np.arange(0.70, 0.7951, 0.005), 3)


# ---------------- scores (out-of-fold on development, refit for test) ----------------
def oof_full(fit, pred, Xa=X, Xb=Xte):
    oof = np.zeros(len(Xa))
    for a, b in FOLDS:
        oof[b] = pred(fit(Xa[a], y[a], wn[a], g[a]), Xa[b])
    return oof, pred(fit(Xa, y, wn, g), Xb)


proba = lambda m, Z: m.predict_proba(Z)[:, 1]
SC = {}
SC["Logistic regression"] = oof_full(lambda Xa, ya, wa, ga: LogisticRegression(max_iter=2000).fit(Xa, ya, sample_weight=wa), proba)


def rw_fit(Xa, ya, wa, ga):
    wr = wa.copy()
    for gg in np.unique(ga):
        for yy in (0, 1):
            m = (ga == gg) & (ya == yy)
            if wa[m].sum() > 0:
                wr[m] *= (wa[ga == gg].sum() * wa[ya == yy].sum() / wa.sum()) / wa[m].sum()
    return LogisticRegression(max_iter=2000).fit(Xa, ya, sample_weight=wr)


SC["Reweighing (Dang et al., 2024)"] = oof_full(rw_fit, proba)


def dir_transform(Xq, gq, Xref, gref):
    Xr = Xq.astype(float).copy()
    for j in range(Xq.shape[1]):
        pooled = np.sort(Xref[:, j])
        for gg in np.unique(gq):
            ref = Xref[gref == gg, j] if (gref == gg).any() else Xref[:, j]
            vals, counts = np.unique(ref, return_counts=True); q = (np.cumsum(counts) - counts / 2) / counts.sum()
            m = gq == gg; Xr[m, j] = np.quantile(pooled, np.interp(Xq[m, j], vals, q, left=0, right=1))
    return Xr


oof = np.zeros(len(X))
for a, b in FOLDS:
    m = LogisticRegression(max_iter=2000).fit(dir_transform(X[a], g[a], X[a], g[a]), y[a], sample_weight=wn[a])
    oof[b] = proba(m, dir_transform(X[b], g[b], X[a], g[a]))
m = LogisticRegression(max_iter=2000).fit(dir_transform(X, g, X, g), y, sample_weight=wn)
SC["Disparate impact remover (Dang et al., 2024)"] = (oof, proba(m, dir_transform(Xte, gte, X, g)))


def khan_fit(Xa, ya, wa, ga, lam=5.0):
    Xs = np.column_stack([np.ones(len(Xa)), Xa]); G = pd.get_dummies(ga).to_numpy(float)

    def loss(b):
        pr = expit(Xs @ b)
        ll = -np.sum(wa * (ya * np.log(pr + 1e-12) + (1 - ya) * np.log(1 - pr + 1e-12))) / wa.sum(); pen = 0.0
        for yy in (0, 1):
            m = ya == yy
            means = (G[m] * (wa[m] * pr[m])[:, None]).sum(0) / ((G[m] * wa[m][:, None]).sum(0) + 1e-12)
            pen += np.sum((means - np.average(pr[m], weights=wa[m])) ** 2)
        return ll + lam * pen
    return minimize(loss, np.zeros(Xs.shape[1]), method="L-BFGS-B").x


SC["Fairness-regularised model (Khan et al., 2026)"] = oof_full(khan_fit, lambda b, Z: expit(np.column_stack([np.ones(len(Z)), Z]) @ b))

s = pickle.load(open(f"out/dafsI_scores{SUF}.pkl", "rb")); PROP = {"DAFS-I (proposed)": (s["oof"], s["pte"])}
if SCALE == "anx":      # GAD-7 primary: items-only score with the same cut-off procedure (09_proposed.py, A3)
    src = open("code/09_proposed.py").read().split("rows = []")[0]; ns = {"__name__": "lib"}
    sys.argv = ["x", "yA", "0.05"]; exec(src, ns); o, p, _ = ns["oof_and_test"](False)
    PROP = {"Items-only score + cross-fitted EO cut-offs (proposed)": (o, p), **PROP}


# ---------------- rule families (development only) ----------------
def dev_metrics(Q):
    """Q: rules x n development decisions -> BAcc and AOD across the mutually exclusive training groups."""
    pos, neg = w * y, w * (1 - y)
    tpr = {gg: Q[:, g == gg] @ pos[g == gg] / pos[g == gg].sum() for gg in GROUPS}
    fpr = {gg: Q[:, g == gg] @ neg[g == gg] / neg[g == gg].sum() for gg in GROUPS}
    aod = np.maximum.reduce([0.5 * (np.abs(tpr[gg] - tpr["none"]) + np.abs(fpr[gg] - fpr["none"])) for gg in GROUPS if gg != "none"])
    bacc = (Q @ pos / pos.sum() + 1 - Q @ neg / neg.sum()) / 2
    return bacc, aod


def eo_family(oof):
    Fs, Ts = np.linspace(0.01, 0.45, 89), np.linspace(0.3, 0.95, 66); thr = {}
    for gg in GROUPS:
        m = g == gg; ss, tp, fp = roc_pts(oof[m], y[m], w[m])
        idx = np.concatenate([np.argmin(np.abs(tp - T)[None, :] + np.abs(fp[None, :] - Fs[:, None]), axis=1) for T in Ts])
        thr[gg] = ss[idx]
    return [dict(zip(GROUPS, v)) for v in zip(*[thr[gg] for gg in GROUPS])]


def psta_family(oof):
    rules = []
    for s_ in np.linspace(0.40, 0.97, 115):
        r = {}
        for gg in GROUPS:
            m = (g == gg) & (y == 1); ss, ww = oof[m], w[m]; o = np.argsort(-ss); cum = np.cumsum(ww[o]) / ww.sum()
            r[gg] = ss[o][min(np.searchsorted(cum, s_), len(o) - 1)]
        rules.append(r)
    return rules


def single_family(oof):
    return [{gg: t for gg in GROUPS} for t in np.unique(np.quantile(oof, np.linspace(0.30, 0.995, 280)))]


def apply(rule, sc, grp):
    t = np.array([rule[x] for x in grp]); return (sc >= t).astype(float)


def select(rules, oof):
    Q = np.array([apply(r, oof, g) for r in rules]); bacc, aod = dev_metrics(Q); out = {}
    for b in LEVELS:
        ok = np.flatnonzero(np.abs(bacc - b) <= BAND / 2 + 1e-12) if BAND else np.flatnonzero(bacc >= b - 1e-12)
        if len(ok):
            k = ok[np.lexsort((-bacc[ok], aod[ok]))[0]]; out[b] = (rules[k], bacc[k], aod[k])
    return out


FAM = {}
for name, (o, p) in PROP.items():
    FAM[name] = (select(eo_family(o), o), p)
FAM["PSTA (Dang et al., 2024)"] = (select(psta_family(SC["Logistic regression"][0]), SC["Logistic regression"][0]), SC["Logistic regression"][1])
for name in ["Reweighing (Dang et al., 2024)", "Disparate impact remover (Dang et al., 2024)", "Fairness-regularised model (Khan et al., 2026)", "Logistic regression"]:
    FAM[name] = (select(single_family(SC[name][0]), SC[name][0]), SC[name][1])

# ---------------- test evaluation with shared Rao-Wu replicates ----------------
yt = te.yA.to_numpy(float)
masks = [(te.disab == 0).to_numpy()] + [(te[f"d_{k}"] == 1).to_numpy() for k in DOMS]
rng = np.random.default_rng(842); B = 500
W = np.vstack([te.wt_pooled.to_numpy()] + [bootstrap_weights(te, rng) for _ in range(B)])   # row 0 = point estimate


def test_metrics(Q):
    """Q: rules x n -> arrays (rules x (B+1)) of BAcc and AOD over overlapping domains vs no disability."""
    P, N = W * yt, W * (1 - yt)
    tp = [(Q * m) @ P.T / (P * m).sum(1) for m in masks]; fp = [(Q * m) @ N.T / (N * m).sum(1) for m in masks]
    aod = np.maximum.reduce([0.5 * (np.abs(tp[k] - tp[0]) + np.abs(fp[k] - fp[0])) for k in range(1, 7)])
    bacc = (Q @ P.T / P.sum(1) + 1 - Q @ N.T / N.sum(1)) / 2
    return bacc, aod


RES, rows = {}, []
for name, (sel, pte) in FAM.items():
    lv = sorted(sel); Q = np.array([apply(sel[b][0], pte, gte) for b in lv]); bacc, aod = test_metrics(Q)
    RES[name] = {b: (bacc[i], aod[i]) for i, b in enumerate(lv)}
    for i, b in enumerate(lv):
        rows.append(dict(method=name, level=b, dev_BAcc=sel[b][1], dev_AOD=sel[b][2], BAcc=bacc[i, 0], AOD=aod[i, 0],
                         AOD_lo=np.quantile(aod[i, 1:], .025), AOD_hi=np.quantile(aod[i, 1:], .975)))
pts = pd.DataFrame(rows); pts.to_csv(f"out/frontier_points_{TAG}.csv", index=False)

drows = []
for prop in PROP:
    for comp in FAM:
        if comp == prop: continue
        common = sorted(set(RES[prop]) & set(RES[comp]))
        if not common: continue
        for b in common + ["mean"]:
            ls = common if b == "mean" else [b]
            for k, met in enumerate(["BAcc", "AOD"]):
                diff = np.mean([RES[prop][l][k] - RES[comp][l][k] for l in ls], axis=0)
                drows.append(dict(method=prop, comparator=comp, level=b, n_levels=len(ls), metric=met,
                                  difference=diff[0], lo=np.quantile(diff[1:], .025), hi=np.quantile(diff[1:], .975)))
dd = pd.DataFrame(drows); dd.to_csv(f"out/frontier_diffs_{TAG}.csv", index=False)
pickle.dump({k: {b: v[0][b][0] for b in v[0]} for k, v in FAM.items()}, open(f"out/frontier_rules_{TAG}.pkl", "wb"))
pd.set_option("display.width", 250)
print(pts.round(3).to_string())
print(dd[dd.level == "mean"].round(3).to_string())
print("FRONTIER COMPLETE", TAG)
