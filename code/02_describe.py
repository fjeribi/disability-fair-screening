"""Stage 1b: domain sizes and weighted rates per split (Table 1 draft)."""
import numpy as np, pandas as pd
df = pd.read_parquet("data/proc/nhis_analytic.parquet")
dom = ["vision","hearing","mobility","cognition","selfcare","communication"]
def wmean(x, w):
    m = x.notna(); return np.nan if m.sum()==0 else np.average(x[m], weights=w[m])
rows=[]
for split in ["train","test"]:
    s = df[df.split==split]
    groups = {"No disability": s.disab==0, "Any disability": s.disab==1}
    groups.update({d: s[f"d_{d}"]==1 for d in dom})
    groups.update({f"{d} (only domain)": (s[f"d_{d}"]==1)&(s.n_domains==1) for d in dom})
    for g, m in groups.items():
        t = s[m]; w = t.wt_pooled
        rows.append(dict(split=split, group=g, n=len(t),
            phq_cases=int(t.phq_pos.sum()), gad_cases=int(t.gad_pos.sum()),
            pct_phq10=100*wmean(t.phq_pos,w), pct_gad10=100*wmean(t.gad_pos,w),
            pct_depmed=100*wmean(t.depmed,w), pct_anxmed=100*wmean(t.anxmed,w),
            pct_therapy=100*wmean(t.therapy,w), pct_depev=100*wmean(t.depev,w),
            pct_anxev=100*wmean(t.anxev,w),
            mean_age=wmean(t.age,w)))
out = pd.DataFrame(rows)
out.to_csv("out/table1_domain_counts.csv", index=False)
pd.set_option("display.width",250); pd.set_option("display.max_columns",20)
print(out.round(1).to_string(index=False))
# overlap of domains among disabled (train)
s=df[(df.split=="train")&(df.disab==1)]
print("\nn_domains among disabled (train):", s.n_domains.value_counts().sort_index().to_dict())
