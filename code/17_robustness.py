"""Stage 7b: robustness of the main finding (PHQ-8 >= 10 false-positive gap, latent-class reference).
R1 current-only indicators (drop lifetime diagnosis): phq_pos, wg_dep, depmed, therapy
R2 single-domain groups (only one functional domain)
R3 graded exposure: 'some difficulty' only (no 'a lot'/'cannot' in any domain) per domain -> dose-response
R4 2025 test year (main analysis used 2019+2022)
"""
import sys; sys.path.insert(0, 'code')
import numpy as np, pandas as pd
src = open("code/06_latent_class.py").read().split("tr = df[df.split")[0]
ns = {}; exec(src, ns); df, fit_group = ns["df"], ns["fit_group"]
DOMS = ["vision","hearing","mobility","cognition","selfcare","communication"]
MAIN = ["phq_pos","wg_dep","depmed","depev","therapy"]; CUR = ["phq_pos","wg_dep","depmed","therapy"]
rows = []
def add(check, group, t, ind):
    if len(t) < 120: return
    r = fit_group(t, ind); rows.append(dict(check=check, group=group, n=r["n"], FPR=100*(1-r["spec"]), sens=100*r["sens"]))
for split, lab in [("train","R0 main (2019+2022)"), ("test","R4 test year 2025")]:
    s = df[df.split==split]
    add(lab, "none", s[s.disab==0], MAIN)
    for k in DOMS: add(lab, k, s[s[f"d_{k}"]==1], MAIN)
tr = df[df.split=="train"]
add("R1 current-only indicators", "none", tr[tr.disab==0], CUR)
for k in DOMS: add("R1 current-only indicators", k, tr[tr[f"d_{k}"]==1], CUR)
for k in DOMS: add("R2 single-domain only", k, tr[(tr[f"d_{k}"]==1)&(tr.n_domains==1)], MAIN)
acc = tr[(tr.uninsured == 0) & (tr.usual_place == 1)]
add("R8 equal access (insured, usual care)", "none", acc[acc.disab==0], MAIN)
for k in DOMS: add("R8 equal access (insured, usual care)", k, acc[acc[f"d_{k}"]==1], MAIN)
lvl = [f"lvl_{k}" for k in DOMS]
some_only = (tr[lvl].max(axis=1)==1)
add("R3 'some difficulty' only", "none (no difficulty at all)", tr[tr[lvl].max(axis=1)==0], MAIN)
for k in DOMS: add("R3 'some difficulty' only", k, tr[some_only & (tr[f"lvl_{k}"]==1)], MAIN)
r = pd.DataFrame(rows); r.to_csv("out/robustness_fpr.csv", index=False)
print(r.pivot_table(index="group", columns="check", values="FPR").round(1).to_string())
print("\nn per cell:\n", r.pivot_table(index="group", columns="check", values="n").to_string())
