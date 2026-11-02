"""Calibration of the DAFS-I score on NHIS 2025: survey-weighted expected calibration error (10 bins) and Brier score
for adults without disability and in each overlapping domain. Output: out/dafsI_calibration.csv"""
import pickle, numpy as np, pandas as pd
s = np.asarray(pickle.load(open("out/pred_dafsI.pkl", "rb"))["DAFS-I"][0], float)
d = pd.read_parquet("data/proc/ml_dep.parquet"); te = d[d.split == "test"].reset_index(drop=True)
y, w = te.yA.to_numpy(float), te.wt_pooled.to_numpy(float)
def ece(m):
    b = np.minimum((s[m] * 10).astype(int), 9); yy, ss, ww = y[m], s[m], w[m]; e = 0.0
    for k in range(10):
        j = b == k
        if j.any(): e += ww[j].sum() * abs(np.average(yy[j], weights=ww[j]) - np.average(ss[j], weights=ww[j]))
    return e / ww.sum()
rows = []
groups = {"No disability": te.disab.to_numpy() == 0}
groups.update({k: te[f"d_{k}"].to_numpy() == 1 for k in ["cognition", "selfcare", "communication", "mobility", "vision", "hearing"]})
for g, m in groups.items():
    rows.append(dict(group=g, n=int(m.sum()), ECE=ece(m), Brier=np.average((s[m] - y[m]) ** 2, weights=w[m])))
out = pd.DataFrame(rows); out.to_csv("out/dafsI_calibration.csv", index=False); print(out.round(3).to_string())
