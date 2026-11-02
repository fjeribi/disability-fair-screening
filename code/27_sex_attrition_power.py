"""Stage 7e: sex-stratified and non-response-weighted robustness of the main finding, and the minimum
detectable difference (sensitivity power analysis) of the audit.

R5 women only, R6 men only: latent-class false-positive rate of PHQ-8 >= 10 per domain (2019+2022).
R7 non-response weighted: survey weight divided by the estimated probability of being analysed
    (all five latent-class indicators observed), from a logistic model on age group, sex, income-to-poverty
    category, race and ethnicity, education, survey year and the six graded domain levels.
Power: minimum detectable difference (two-sided alpha = 0.05, power = 0.80) between each domain's
    false-positive rate and the reference group's: MDD = (z_0.975 + z_0.80) * SD of the paired bootstrap
    difference (domain minus reference false-positive rate within the same 200 Rao-Wu replicates,
    out/corrected_lca_bootstrap.csv), so the covariance of the two rates is taken into account.
Outputs: out/robustness_sex_attrition.csv, out/power_mdd.csv
"""
import sys; sys.path.insert(0, 'code')
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

src = open("code/06_latent_class.py").read().split("tr = df[df.split")[0]
ns = {}; exec(src, ns); df, fit_group = ns["df"], ns["fit_group"]
DOMS = ["vision", "hearing", "mobility", "cognition", "selfcare", "communication"]
MAIN = ["phq_pos", "wg_dep", "depmed", "depev", "therapy"]
tr = df[(df.split == "train") & df.disab.notna()].copy()

rows = []
def add(check, group, t, ind=MAIN):
    if t.dropna(subset=ind).shape[0] < 120: return
    r = fit_group(t, ind)
    rows.append(dict(check=check, group=group, n=r["n"], FPR=100 * (1 - r["spec"]), sens=100 * r["sens"]))

for lab, m in [("R5 women", tr.female == 1), ("R6 men", tr.female == 0)]:
    s = tr[m]
    add(lab, "none", s[s.disab == 0])
    for k in DOMS: add(lab, k, s[s[f"d_{k}"] == 1])

# R7: inverse probability of being analysed
inc = tr[MAIN].notna().all(axis=1).astype(int)
Z = pd.DataFrame({"age": pd.cut(tr.age, [17, 29, 44, 64, 74, 120]).astype(str),
                  "female": tr.female.fillna(-1).astype(int).astype(str),
                  "inc": tr.income_ratio_cat.astype(str), "race": tr.race_eth.astype(str),
                  "educ": tr.educ.fillna(-1).astype(int).astype(str), "year": tr.year.astype(str)})
for k in DOMS: Z[k] = tr[f"lvl_{k}"].fillna(-1).astype(int).astype(str)
X = pd.get_dummies(Z, drop_first=True).astype(float)
ph = LogisticRegression(C=1e6, max_iter=5000).fit(X, inc).predict_proba(X)[:, 1]
tw = tr.copy(); tw["wt_pooled"] = tr.wt_pooled / ph
print("analysed share %.3f; inverse-probability factor range %.2f-%.2f" % (inc.mean(), (1 / ph[inc == 1]).min(), (1 / ph[inc == 1]).max()))
add("R7 non-response weighted", "none", tw[tw.disab == 0])
for k in DOMS: add("R7 non-response weighted", k, tw[tw[f"d_{k}"] == 1])

r = pd.DataFrame(rows); r.to_csv("out/robustness_sex_attrition.csv", index=False)
print(r.pivot_table(index="group", columns="check", values="FPR").round(1).to_string())
print(r.pivot_table(index="group", columns="check", values="n").to_string())

# minimum detectable difference of the audit
lca = pd.read_csv("out/corrected_lca_intervals.csv"); lca = lca[(lca.scale == "dep") & (lca.survey == "NHIS")].set_index("group")
bs = pd.read_csv("out/corrected_lca_bootstrap.csv"); bs = bs[(bs.scale == "dep") & (bs.survey == "NHIS")]
fp_b = (100 * (1 - bs.spec)).groupby([bs.replicate, bs.group]).first().unstack()     # replicate x group
fpr = lambda g: 100 * (1 - lca.loc[g, "spec"])
k = norm.ppf(0.975) + norm.ppf(0.80)
sd = lambda g: float((fp_b[g] - fp_b["No disability"]).std(ddof=1))
p = pd.DataFrame([dict(domain=g, n=int(lca.loc[g, "n"]), FPR=fpr(g), gap=fpr(g) - fpr("No disability"),
                       SE_gap=sd(g), MDD=k * sd(g), replicates=int(fp_b[g].notna().sum())) for g in DOMS])
p["gap_over_MDD"] = p.gap / p.MDD
p.to_csv("out/power_mdd.csv", index=False)
print(p.round(2).to_string(index=False))
