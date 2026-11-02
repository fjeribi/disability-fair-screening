"""Stage 2c: latent-class reference standard (no gold standard needed).

Two-class, survey-weighted latent class model fitted separately in each disability domain and in
adults without disability (NHIS 2019+2022). Binary indicators of depression:
  phq_pos  PHQ-8 >= 10 (the screener under evaluation)
  wg_dep   Washington Group affect item: depressed daily/weekly AND level 'a lot' or 'in between'
  depmed   currently takes prescription medication for depression
  depev    ever told had depression (lifetime diagnosis)
  therapy  counselling/therapy from a mental-health professional, past 12 months
Group-specific sensitivity = P(phq_pos=1 | case class); specificity = P(phq_pos=0 | non-case class).
Anxiety model uses gad_pos, wg_anx, anxmed, anxev, therapy.
Bootstrap over PSUs (within strata) gives 95% CIs.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
df = pd.read_parquet("data/proc/nhis_analytic.parquet")
df["wg_dep"] = (df.depfreq.isin([1, 2]) & df.deplevel.isin([2, 3])).astype(float).where(df.depfreq.notna())
df["wg_anx"] = (df.anxfreq.isin([1, 2]) & df.anxlevel.isin([2, 3])).astype(float).where(df.anxfreq.notna())
# deplevel/anxlevel asked only of those not answering 'never'; never -> 0
df.loc[df.depfreq == 5, "wg_dep"] = 0.0
df.loc[df.anxfreq == 5, "wg_anx"] = 0.0
df.loc[df.depfreq.isin([1,2]) & df.deplevel.isna(), "wg_dep"] = np.nan
df.loc[df.anxfreq.isin([1,2]) & df.anxlevel.isna(), "wg_anx"] = np.nan
IND = {"dep": ["phq_pos", "wg_dep", "depmed", "depev", "therapy"],
       "anx": ["gad_pos", "wg_anx", "anxmed", "anxev", "therapy"]}
DOMS = ["vision", "hearing", "mobility", "cognition", "selfcare", "communication"]


from statutils import lca2


def fit_group(t, ind):
    t = t.dropna(subset=ind)
    pi, p = lca2(t[ind].to_numpy(), t.wt_pooled.to_numpy())
    return dict(n=len(t), latent_prev=pi, sens=p[1][0], spec=1 - p[0][0],
                screen_pos=np.average(t[ind[0]], weights=t.wt_pooled))


def boot(t, ind, B=200):
    out = []
    psus = t.groupby("stratum")["psu"].unique()
    for _ in range(B):
        pick = np.concatenate([rng.choice(v, len(v), replace=True) for v in psus.values])
        bt = t.set_index("psu").loc[pick].reset_index()
        out.append(fit_group(bt, ind))
    return pd.DataFrame(out)


tr = df[df.split == "train"]
rows = []
for scale, ind in IND.items():
    groups = {"No disability": tr[tr.disab == 0], "Any disability": tr[tr.disab == 1]}
    groups.update({d: tr[tr[f"d_{d}"] == 1] for d in DOMS})
    for g, t in groups.items():
        est = fit_group(t, ind)
        b = boot(t, ind, B=200 if g != "No disability" else 100)
        for m in ["sens", "spec", "latent_prev"]:
            est[f"{m}_lo"], est[f"{m}_hi"] = np.percentile(b[m], [2.5, 97.5])
        rows.append(dict(scale=scale, group=g, **est))
        print(scale, g, {k: round(v, 3) for k, v in est.items() if k in ("n", "latent_prev", "sens", "spec")}, flush=True)
res = pd.DataFrame(rows)
res.to_csv("out/lca_accuracy.csv", index=False)
