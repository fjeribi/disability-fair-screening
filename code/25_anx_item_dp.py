"""GAD-7 extension, step 2: item-level differential prediction (same model as 10_explain_calibrate.py part 3)
and the pre-specified selection rule for DAFS-I adjustments: items with ratio of odds ratios < 1 and p < 0.05
in at least three domains (the rule that selects PHQ items 4 and 7 for depression)."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, "code"); from fairmetrics import DOMS
d = pd.read_parquet("data/proc/ml_anx.parquet"); tr = d[d.split == "train"]
ITEMS = [f"gad7{i}" for i in range(1, 8)]
NAMES = {"gad71": "1 nervous", "gad72": "2 can't stop worrying", "gad73": "3 worrying too much", "gad74": "4 trouble relaxing",
         "gad75": "5 restless", "gad76": "6 irritable", "gad77": "7 afraid"}
res = []
for k in DOMS:
    s = tr[(tr[f"d_{k}"] == 1) | (tr.disab == 0)].copy(); s["D"] = (s[f"d_{k}"] == 1).astype(float)
    X = s[ITEMS].copy(); X["D"] = s.D
    for c in ITEMS: X[f"{c}xD"] = s[c] * s.D
    fit = sm.GLM(s.yA, sm.add_constant(X), family=sm.families.Binomial(), freq_weights=s.wt_pooled / s.wt_pooled.mean()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(s.psu)[0]})
    for c in ITEMS:
        b, se = fit.params[f"{c}xD"], fit.bse[f"{c}xD"]
        res.append(dict(domain=k, item=NAMES[c], code=c, OR_ratio=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=fit.pvalues[f"{c}xD"]))
dp = pd.DataFrame(res); dp.to_csv("out/item_differential_prediction_gad.csv", index=False)
piv = dp.assign(cell=dp.apply(lambda r: f"{r.OR_ratio:.2f}{'*' if r.p < 0.05 else ''}", axis=1)).pivot(index="item", columns="domain", values="cell")
print(piv.to_string())
sel = dp[(dp.OR_ratio < 1) & (dp.p < 0.05)].groupby("code").size()
chosen = sorted(sel[sel >= 3].index.tolist()); print("\nsignificant-domain counts:", sel.to_dict(), "\nselected items:", chosen)
pd.Series(chosen).to_csv("out/gad_selected_items.csv", index=False, header=["item"])
