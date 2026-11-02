"""Table 13: fairness across operating points, narrow accuracy bands (summary of 33_frontier.py with BAND=0.005), added to out/paper_tables.json as key F."""
import json, numpy as np, pandas as pd
rows = []
for sc, lab, prop in [("dep", "PHQ-8", "DAFS-I (proposed)"), ("anx", "GAD-7", "Items-only score + cross-fitted EO cut-offs (proposed)")]:
    p = pd.read_csv(f"out/frontier_points_{sc}_band.csv"); d = pd.read_csv(f"out/frontier_diffs_{sc}_band.csv")
    for comp in ["PSTA (Dang et al., 2024)", "Reweighing (Dang et al., 2024)", "Disparate impact remover (Dang et al., 2024)", "Fairness-regularised model (Khan et al., 2026)"]:
        a = p[p.method == prop].set_index("level"); b = p[p.method == comp].set_index("level"); c = a.index.intersection(b.index)
        x = d[(d.method == prop) & (d.comparator == comp) & (d.level == "mean")].set_index("metric")
        f = lambda m: f"{x.loc[m, 'difference']:+.3f} ({x.loc[m, 'lo']:+.3f} to {x.loc[m, 'hi']:+.3f})"
        rows.append([lab, comp, f"{len(c)}", f"{int((a.AOD[c] < b.AOD[c]).sum())}", f"{int((a.dev_AOD[c] < b.dev_AOD[c]).sum())}",
                     f(  "AOD"), f("BAcc")])
cols = ["Scale", "Comparator", "Accuracy levels", "Levels with lower AOD, 2025", "Levels with lower AOD, development", "Mean Δ AOD (95% CI)", "Mean Δ balanced accuracy (95% CI)"]
md = "| " + " | ".join(cols) + " |\n|:--|" + "--:|" * (len(cols) - 1) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows)
T = json.load(open("out/paper_tables.json")); T["F"] = md; json.dump(T, open("out/paper_tables.json", "w"), ensure_ascii=False, indent=1)
open("out/frontier_table.md", "w").write(md + "\n"); print(md)
