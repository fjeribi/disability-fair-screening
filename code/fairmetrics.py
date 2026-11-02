"""Survey-weighted performance and fairness metrics across overlapping disability domains."""
import numpy as np
from sklearn.metrics import roc_auc_score

DOMS = ["vision", "hearing", "mobility", "cognition", "selfcare", "communication"]


def wrate(mask, w):
    return np.sum(w[mask]) / np.sum(w) if np.sum(w) > 0 else np.nan


def group_rates(y, yhat, w):
    y, q, w = np.asarray(y,float), np.asarray(yhat,float), np.asarray(w,float)
    if not np.isfinite(q).all() or ((q < -1e-10)|(q>1+1e-10)).any(): raise ValueError("Decisions must be binary or intended decision probabilities")
    q = np.clip(q, 0, 1)
    return (np.sum(w*y*q)/np.sum(w*y), np.sum(w*(1-y)*q)/np.sum(w*(1-y)), np.sum(w*q)/np.sum(w))


def evaluate(df, score, yhat, ycol, wcol="wt_pooled"):
    """df: test frame with d_<domain> flags and disab; score: continuous; yhat: 0/1."""
    y, w = df[ycol].to_numpy(), df[wcol].to_numpy()
    out = {"AUC": roc_auc_score(y, score, sample_weight=w)}
    tpr, fpr, pr = group_rates(y, yhat, w)
    prec = np.sum(w*y*yhat) / max(np.sum(w*yhat), 1e-12)
    out.update(BAcc=(tpr + 1 - fpr) / 2, F1=2 * prec * tpr / (prec + tpr), TPR=tpr, FPR=fpr)
    groups = {"none": (df.disab == 0).to_numpy()}
    groups.update({d: (df[f"d_{d}"] == 1).to_numpy() for d in DOMS})
    rates = {g: group_rates(y[m], yhat[m], w[m]) for g, m in groups.items()}
    ref = rates["none"]
    dt = {g: abs(r[0] - ref[0]) for g, r in rates.items() if g != "none"}
    df_ = {g: abs(r[1] - ref[1]) for g, r in rates.items() if g != "none"}
    dp = {g: abs(r[2] - ref[2]) for g, r in rates.items() if g != "none"}
    out["EOD"] = max(dt.values())
    out["FPRgap"] = max(df_.values())
    out["AOD"] = max(0.5 * (dt[g] + df_[g]) for g in dt)
    out["DPD"] = max(dp.values())
    for g, r in rates.items():
        out[f"TPR_{g}"], out[f"FPR_{g}"] = r[0], r[1]
    return out

