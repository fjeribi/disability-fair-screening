"""Latent-class sensitivity to the conditional-independence assumption (added in the second revision).
Indicators can stay associated within a class (for example, a lifetime diagnosis and current medication).
The PHQ-8 >= 10 audit is refitted (NHIS 2019+2022, survey-weighted, five random starts) under alternative
indicator specifications:
  main          : PHQ-8 >= 10 + WG affect item + medication + lifetime diagnosis + counselling
  drop_<x>      : the main model without one external indicator (four specifications)
  combined_dx_rx: diagnosis and medication merged into one indicator (diagnosis or medication), which
                  absorbs any within-class dependence between the two
For each specification: false-positive rate per domain and in the reference group, the gap, the Spearman
correlation of the domain ranking with the main model, and convergence of the selected solution.
Output: out/lca_sensitivity.csv, out/lca_sensitivity_summary.csv
"""
from pathlib import Path
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
os.chdir(Path(__file__).resolve().parents[1]); sys.path.insert(0, "code")
from statutils import lca2
src = Path("code/06_latent_class.py").read_text(encoding="utf-8").split("tr = df[df.split")[0]; ns = {}
exec(compile(src, "inspected_LCA_definitions", "exec"), ns); df, DOMS = ns["df"], ns["DOMS"]
t = df[df.split == "train"].copy(); t["dx_or_rx"] = ((t.depev == 1) | (t.depmed == 1)).astype(float)
t.loc[t.depev.isna() | t.depmed.isna(), "dx_or_rx"] = np.nan
ext = ["wg_dep", "depmed", "depev", "therapy"]
SPECS = {"main": ["phq_pos"] + ext}
SPECS.update({f"drop_{x}": ["phq_pos"] + [e for e in ext if e != x] for x in ext})
SPECS["combined_dx_rx"] = ["phq_pos", "wg_dep", "dx_or_rx", "therapy"]
groups = {"No disability": t.disab == 0}; groups.update({k: t["d_" + k] == 1 for k in DOMS})
rows = []
for spec, ind in SPECS.items():
    for g, m in groups.items():
        tt = t.loc[m].dropna(subset=ind)
        pi, p, meta = lca2(tt[ind].to_numpy(), tt.wt_pooled.to_numpy(), diagnostics=True)
        boundary = bool(np.any((p < 1e-4) | (p > 1 - 1e-4)))
        rows.append(dict(spec=spec, group=g, n=len(tt), latent_prev=pi, FPR=100 * p[0, 0], sens=100 * p[1, 0],
                         converged=meta["converged"], iterations=meta["iterations"], boundary_parameter=boundary))
    print(spec, "done", flush=True)
d = pd.DataFrame(rows); d.to_csv("out/lca_sensitivity.csv", index=False)
piv = d.pivot(index="group", columns="spec", values="FPR"); main = piv["main"][DOMS]
summ = []
for spec in SPECS:
    v = piv[spec]; gaps = v[DOMS] - v["No disability"]
    summ.append(dict(spec=spec, FPR_ref=v["No disability"], FPR_min_domain=v[DOMS].min(), FPR_max_domain=v[DOMS].max(),
                     top_domain=v[DOMS].idxmax(), gap_min=gaps.min(), gap_max=gaps.max(),
                     spearman_vs_main=spearmanr(main, v[DOMS])[0], all_converged=bool(d[d.spec == spec].converged.all()),
                     any_boundary=bool(d[d.spec == spec].boundary_parameter.any())))
s = pd.DataFrame(summ); s.to_csv("out/lca_sensitivity_summary.csv", index=False)
pd.set_option("display.width", 220); print(piv.round(1).to_string()); print(s.round(3).to_string())
