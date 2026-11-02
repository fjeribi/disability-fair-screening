"""Stage 8: render manuscript tables (pipe Markdown) directly from result files."""
import json
import re
import pandas as pd

O = "out/"
DOM = ["cognition", "selfcare", "communication", "mobility", "vision", "hearing"]
LAB = {"cognition": "Cognition", "selfcare": "Self-care", "communication": "Communication", "mobility": "Mobility",
       "vision": "Vision", "hearing": "Hearing", "none": "No disability", "No disability": "No disability",
       "Any disability": "Any disability"}


def _clean(x):
    x = str(x).replace(">= ", "≥ ").replace(">=", "≥ ").replace("Delta ", "Δ ")
    return re.sub(r"(?<=\d)-(?=\d)", "–", x)          # en dash in numeric ranges


def md(df):
    df = df.copy()
    df.columns = [_clean(c) for c in df.columns]
    df = df.apply(lambda col: col.map(_clean)) if len(df) else df
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


T = {}
# Table 1: related work
rw = [
    ["Cintron et al. (2023)", "NHIS 2019, PHQ-8", "Age, gender, race, education", "No", "No", "No", "Measurement only"],
    ["De La Rosa et al. (2026)", "NHIS 2019, PHQ-8", "Chronic pain", "No", "No", "No", "Measurement only"],
    ["Katzan et al. (2021)", "Clinic, PHQ-9", "Neurological diagnoses", "No", "No", "No", "Measurement only"],
    ["Delamain et al. (2024)", "Clinical, PHQ-9/GAD-7", "Age", "No", "No", "No", "Measurement only"],
    ["Levis et al. (2019)", "IPD meta-analysis, PHQ-9", "Age, sex, setting", "No", "Yes (interview)", "Yes", "No"],
    ["Lai et al. (2019)", "Addiction scale", "Two groups", "No", "No (model-implied)", "Yes", "No"],
    ["Dang et al. (2024)", "NHANES + 3 cohorts", "Sex, ethnicity, age, income", "No", "No (PHQ-9 label)", "Yes", "Yes"],
    ["Tian et al. (2026)", "Students, multimodal", "Gender, age, level", "No", "No (PHQ/GAD label)", "Yes", "No"],
    ["Khan et al. (2026)", "Interview/speech/text", "Gender, age", "No", "No (PHQ label)", "Yes", "No"],
    ["Adzrago et al. (2026)", "NHIS 2019-2023", "Disability status", "No", "No", "No (prevalence)", "No"],
    ["Meshkat et al. (2025)", "NHANES 2013-2018", "Disability domains", "Yes", "No", "No (association)", "No"],
    ["**This study**", "**NHIS 2019, 2022, 2025; NHANES 2021-2023**", "**Six WG domains**", "**Yes**", "**Yes (latent class + model-implied)**", "**Yes**", "**Yes**"],
]
T[1] = md(pd.DataFrame(rw, columns=["Study", "Data", "Groups compared", "Disability domains", "Criterion beyond the questionnaire", "Accuracy or fairness by group", "Corrective screening rule"]))

# Table 2: sample
t = pd.read_csv(O + "table1_domain_counts.csv")
_d = pd.read_parquet("data/proc/nhis_analytic.parquet"); _d = _d[_d.split == "train"]
import numpy as _np
def _fem(g):
    m = (_d.disab == 0) if g == "No disability" else (_d.disab == 1) if g == "Any disability" else (_d[f"d_{g}"] == 1)
    x = _d[m & _d.female.notna()]; return 100 * _np.average(x.female, weights=x.wt_pooled)
rows = []
for g in ["No disability", "Any disability"] + DOM:
    a = t[(t.split == "train") & (t.group == g)].iloc[0]
    b = t[(t.split == "test") & (t.group == g)].iloc[0]
    rows.append([LAB.get(g, g), f"{a.n:,}", f"{a.mean_age:.1f}", f"{_fem(g):.1f}", f"{a.pct_phq10:.1f}", f"{a.pct_gad10:.1f}",
                 f"{a.pct_depmed:.1f}", f"{a.pct_therapy:.1f}", f"{b.n:,}", f"{b.pct_phq10:.1f}"])
T[2] = md(pd.DataFrame(rows, columns=["Group", "n (2019+2022)", "Mean age", "Female (%)", "PHQ-8 >= 10 (%)", "GAD-7 >= 10 (%)",
                                      "Depression medication (%)", "Counselling (%)", "n (2025)", "PHQ-8 >= 10 (%) 2025"]))

# Table 3: invariance
f = pd.read_csv(O + "invariance_fit.csv")
rows = []
for sc, nm in [("phq", "PHQ-8"), ("gad", "GAD-7")]:
    for dm in DOM:
        x = f[(f.scale == sc) & (f.domain == dm)].set_index("step")
        rows.append([nm, LAB[dm], f"{int(x.loc['configural', 'n_domain']):,}",
                     f"{x.loc['configural', 'cfi.scaled']:.3f}", f"{x.loc['scalar', 'cfi.scaled']:.3f}",
                     f"{x.loc['scalar', 'cfi.scaled'] - x.loc['configural', 'cfi.scaled']:+.3f}",
                     f"{x.loc['configural', 'rmsea.scaled']:.3f}", f"{x.loc['scalar', 'rmsea.scaled']:.3f}"])
T[3] = md(pd.DataFrame(rows, columns=["Scale", "Domain", "n (domain)", "CFI configural", "CFI scalar", "Delta CFI", "RMSEA configural", "RMSEA scalar"]))

# Table 4: classification accuracy of PHQ-8 >= 10
mi = pd.read_csv(O + "model_implied_accuracy.csv")
mi = mi[(mi.scale == "phq") & (mi.model == "scalar") & (mi.latent_cut == "millsap_kwok")]
lca = pd.read_csv(O + "lca_accuracy.csv"); lca = lca[lca.scale == "dep"].set_index("group")
nh = pd.read_csv(O + "nhanes_lca.csv").set_index("group")
rows = []
ref = mi[(mi.domain == "cognition") & (mi.group == "nodisab")].iloc[0]
L = lca.loc["No disability"]
rows.append(["No disability", f"{100*ref.sens:.1f}", f"{100*(1-ref.spec):.1f}", f"{100*L.sens:.1f}",
             f"{100*(1-L.spec):.1f} ({100*(1-L.spec_hi):.1f}-{100*(1-L.spec_lo):.1f})", f"{nh.loc['No disability','FPR']:.1f}"])
for dm in DOM:
    m = mi[(mi.domain == dm) & (mi.group == "domain")].iloc[0]; L = lca.loc[dm]
    rows.append([LAB[dm], f"{100*m.sens:.1f}", f"{100*(1-m.spec):.1f}", f"{100*L.sens:.1f}",
                 f"{100*(1-L.spec):.1f} ({100*(1-L.spec_hi):.1f}-{100*(1-L.spec_lo):.1f})",
                 f"{nh.loc[dm,'FPR']:.1f}" if dm in nh.index else "—"])
T[4] = md(pd.DataFrame(rows, columns=["Group", "Sensitivity, model-implied (%)", "FPR, model-implied (%)",
                                      "Sensitivity, latent class (%)", "FPR, latent class (%) (95% CI)", "FPR, NHANES latent class (%)"]))

# Table 5: superiority (CIs)
ci = pd.read_csv(O + "table_superiority_CI.csv")
order = ["PHQ-8 >= 10 (standard)", "PHQ-2 >= 3", "Logistic regression (unaware)", "XGBoost (unaware)", "Dang: Reweighing (RW)",
         "Dang: Disparate impact remover", "Dang: Calibrated EO post-proc (CPP)", "Dang: PSTA group thresholds",
         "Shipman: fair XGBoost (dual, TPR parity)", "Tian: fair stacking ensemble (FCSE)", "Khan: fairness-regularised model", "DAFS-I (item-informed)"]
nice = {"PHQ-8 >= 10 (standard)": "PHQ-8 >= 10 (standard rule)", "PHQ-2 >= 3": "PHQ-2 >= 3",
        "Logistic regression (unaware)": "Logistic regression", "XGBoost (unaware)": "XGBoost",
        "Dang: Reweighing (RW)": "Reweighing (Dang et al., 2024)", "Dang: Disparate impact remover": "Disparate impact remover (Dang et al., 2024)",
        "Dang: Calibrated EO post-proc (CPP)": "Calibrated EO post-processing (Dang et al., 2024)", "Dang: PSTA group thresholds": "PSTA (Dang et al., 2024)",
        "Shipman: fair XGBoost (dual, TPR parity)": "Fair XGBoost, dual ascent (Shipman and Ushakova, 2026)",
        "Tian: fair stacking ensemble (FCSE)": "Fair stacking ensemble (Tian et al., 2026)",
        "Khan: fairness-regularised model": "Fairness-regularised model (Khan et al., 2026)", "DAFS-I (item-informed)": "**DAFS-I (proposed)**"}
c = ci.set_index("method").loc[order].reset_index()
c["Method"] = c.method.map(nice)
for k in ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD", "DPD"]:
    c[k] = c[k].str.replace("-", "–", regex=False)
c.loc[c.method == "DAFS-I (item-informed)", ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD", "DPD"]] = \
    c.loc[c.method == "DAFS-I (item-informed)", ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD", "DPD"]].map(lambda s: f"**{s}**")
T[5] = md(c[["Method", "AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD", "DPD"]].rename(columns={"BAcc": "Balanced accuracy", "FPRgap": "FPR gap"}))

# Table 6: paired comparisons DAFS-I vs published
pr = pd.read_csv(O + "table_superiority_paired_DAFS-I.csv")
pr = pr[pr.comparator.isin(order[:-1])].copy()
pr["Comparator"] = pr.comparator.map(nice)
def fmt(d, p):
    k = round(p * 300)                                 # bootstrap exceedances out of B = 300
    ps = "< 0.0033" if k == 0 else ("= " + (f"{k/300:.4f}" if k / 300 < 0.01 else f"{k/300:.3f}"))
    return f"{float(d):+.3f} (p {ps})"
pr["Delta AOD"] = [fmt(a, b) for a, b in zip(pr.d_AOD, pr.p_AOD)]
pr["Delta EOD"] = [fmt(a, b) for a, b in zip(pr.d_EOD, pr.p_EOD)]
pr["Delta FPR gap"] = [fmt(a, b) for a, b in zip(pr.d_FPRgap, pr.p_FPRgap)]
pr["Delta balanced accuracy"] = [f"{float(x):+.3f}" for x in pr.d_BAcc]
T[6] = md(pr[["Comparator", "Delta AOD", "Delta EOD", "Delta FPR gap", "Delta balanced accuracy"]])

# Table 9: ablation of DAFS-I components (PHQ-8 and GAD-7)
ab = pd.read_csv(O + "table_ablation.csv")
ab.insert(0, "Scale", "PHQ-8")
ga = pd.read_csv(O + "proposed_yA_anx.csv")
gi = pd.read_csv(O + "table_superiority_CI_anx.csv").set_index("method").loc["DAFS-I (item-informed)"]
gmap = {"A1 unaware score + single threshold": "Items-only score (XGBoost), single cut-off",
        "A2 domain-aware score + single threshold": "Domain-aware score (XGBoost), single cut-off",
        "A3 unaware score + EO thresholds": "Items-only score (XGBoost) + domain EO cut-offs",
        "DAFS (proposed)": "Domain-aware score (XGBoost) + domain EO cut-offs"}
grows = [dict(Scale="GAD-7", configuration=gmap[r.method], **{k: r[k] for k in ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD"]}) for _, r in ga.iterrows()]
grows.append(dict(Scale="GAD-7", configuration="DAFS-I: item-informed score (item 1) + domain EO cut-offs",
                  **{k: float(str(gi[k]).split()[0]) for k in ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD"]}))
ab = pd.concat([ab, pd.DataFrame(grows)], ignore_index=True)
for _c in ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD"]:
    ab[_c] = ab[_c].map(lambda v: f"{float(v):.3f}")
ab = ab.rename(columns={"configuration": "Configuration", "BAcc": "Balanced accuracy", "FPRgap": "FPR gap"})
T[8] = md(ab)

# Table 7: GAD-7 comparison (CIs)
cg = pd.read_csv(O + "table_superiority_CI_anx.csv").set_index("method")
gorder = ["GAD-7 >= 10 (standard)", "GAD-2 >= 3", "Logistic regression (unaware)", "XGBoost (unaware)", "Dang: Reweighing (RW)",
          "Dang: Disparate impact remover", "Dang: Calibrated EO post-proc (CPP)", "Dang: PSTA group thresholds",
          "Shipman: fair XGBoost (dual, TPR parity)", "Tian: fair stacking ensemble (FCSE)", "Khan: fairness-regularised model",
          "A3 unaware score + EO thresholds", "DAFS-I (item-informed)"]
gnice = dict(nice); gnice.update({"GAD-7 >= 10 (standard)": "GAD-7 >= 10 (standard rule)", "GAD-2 >= 3": "GAD-2 >= 3",
          "A3 unaware score + EO thresholds": "**Cross-fitted domain EO cut-offs on items-only score (proposed)**", "DAFS-I (item-informed)": "DAFS-I (item 1 adjustment)"})
g = cg.loc[gorder].reset_index()
g["Method"] = g.method.map(gnice)
for k in ["AUC", "BAcc", "EOD", "FPRgap", "AOD"]:
    g[k] = g[k].str.replace("-", "–", regex=False)
m = g.method == "A3 unaware score + EO thresholds"
g.loc[m, ["AUC", "BAcc", "EOD", "FPRgap", "AOD"]] = g.loc[m, ["AUC", "BAcc", "EOD", "FPRgap", "AOD"]].map(lambda s: f"**{s}**")
T["G"] = md(g[["Method", "AUC", "BAcc", "EOD", "FPRgap", "AOD"]].rename(columns={"BAcc": "Balanced accuracy", "FPRgap": "FPR gap"}))

# Table 8: gap decomposition
dec = pd.read_csv(O + "gap_decomposition.csv").set_index("domain").loc[DOM].reset_index()
T[7] = md(pd.DataFrame({"Domain": dec.domain.map(LAB), "Observed gap": (100 * dec.observed_gap).round(1),
                        "Base rate": (100 * dec.base_rate).round(1), "Item-level predictive bias": (100 * dec.item_predictive).round(1),
                        "Access to care": (100 * dec.access).round(1)}))

# Table 10: scoring sheet and domain cut-offs
pts = pd.read_csv(O + "scoring_sheet_points.csv").set_index("component").points_per_response_level
cut = pd.read_csv(O + "scoring_sheet_cutoffs.csv").set_index("group").cutoff_points
names = {"phq81": "1. Little interest or pleasure", "phq82": "2. Feeling down, depressed or hopeless", "phq83": "3. Sleep problems",
         "phq84": "4. Feeling tired or little energy", "phq85": "5. Poor appetite or overeating", "phq86": "6. Feeling bad about yourself",
         "phq87": "7. Trouble concentrating", "phq88": "8. Moving or speaking slowly / restless",
         "tired_x_disab": "Adjustment, item 4, any WG disability", "conc_x_disab": "Adjustment, item 7, any WG disability"}
sheet = pd.DataFrame({"Component": [names[k] for k in pts.index], "Points per response level (0-3)": pts.values})
T[9] = md(sheet)
T["9b"] = md(pd.DataFrame({"Group": [LAB[g] for g in ["none"] + DOM], "Screen positive at (points)": [f">= {int(cut[g])}" for g in ["none"] + DOM]}))

# Table 11: robustness
r = pd.concat([pd.read_csv(O + "robustness_fpr.csv"), pd.read_csv(O + "robustness_sex_attrition.csv")])
p = r.pivot_table(index="group", columns="check", values="FPR").round(1)
p = p.reindex(["none", "none (no difficulty at all)"] + DOM)
p.index = [LAB.get(i, "No difficulty in any domain") for i in p.index]
p = p.fillna("—").reset_index().rename(columns={"index": "Group", "R0 main (2019+2022)": "Main (2019+2022)",
                                              "R1 current-only indicators": "Current-only indicators", "R2 single-domain only": "Single domain only",
                                              "R3 'some difficulty' only": "'Some difficulty' only", "R4 test year 2025": "Test year 2025",
                                              "R5 women": "Women", "R6 men": "Men", "R7 non-response weighted": "Non-response weighted"})
T[10] = md(p)

# Table 12: three-way conformal screening
cf = pd.concat([pd.read_csv(O + "conformal_three_way_a0.10.csv"), pd.read_csv(O + "conformal_three_way_a0.20.csv")])
rows = []
for a in [0.10, 0.20]:
    for g in ["none"] + DOM:
        x = cf[(cf.alpha.round(2) == a) & (cf.group == g)].set_index("method")
        rows.append([f"{a:.2f}", LAB[g], f"{100*x.loc['Marginal conformal','coverage']:.1f}",
                     f"{100*x.loc['Mondrian conformal (proposed)','coverage']:.1f}",
                     f"{100*x.loc['Mondrian conformal (proposed)','refer_rate']:.1f}",
                     f"{100*x.loc['Mondrian conformal (proposed)','acc_decided']:.1f}"])
T[11] = md(pd.DataFrame(rows, columns=["alpha", "Group", "Coverage, single calibration (%)", "Coverage, domain-conditional (%)",
                                       "Refer rate, domain-conditional (%)", "Accuracy among decided, domain-conditional (%)"]))
# Write all paper tables, numbered as in the manuscript, to out/paper_tables.md (and JSON for programmatic use)
PAPER_NO = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "G": "7", "7": "8", "8": "9",
            "9": "10", "9b": "10 (continued)", "10": "11", "11": "12"}
json.dump({str(k): v for k, v in T.items()}, open("out/paper_tables.json", "w"))
with open("out/paper_tables.md", "w") as fh:
    fh.write("# Tables of the paper (generated by code/22_paper_tables.py)\n\n")
    for k, no in PAPER_NO.items():
        fh.write(f"## Table {no}\n\n{T[k] if k in T else T[int(k)]}\n\n")
print("written out/paper_tables.md")
