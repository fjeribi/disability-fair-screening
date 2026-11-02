"""Render every manuscript table (pipe Markdown, manuscript numbering) from the corrected outputs.
Keys follow the manuscript placeholders: 1, 2, 3, 4, 5, 6, G (Table 7), 7 (Table 8), 8 (Table 9),
9 and 9b (Table 10), 10 (Table 11), 11 (Table 12). Writes out/paper_tables.json and out/paper_tables.md."""
import json, re
import numpy as np, pandas as pd

O = "out/"
DOM = ["cognition", "selfcare", "communication", "mobility", "vision", "hearing"]
LAB = {"cognition": "Cognition", "selfcare": "Self-care", "communication": "Communication", "mobility": "Mobility",
       "vision": "Vision", "hearing": "Hearing", "none": "No disability", "No disability": "No disability",
       "Any disability": "Any disability"}
src = open("code/22_paper_tables.py").read()
ns = {}; exec(src[:src.index("# Table 2: sample")], ns)          # md(), _clean() and Table 1 (related work)
md, T = ns["md"], {1: ns["T"][1]}
f1 = lambda x: f"{x:.1f}"
f3 = lambda x: f"{x:.3f}"
ci3 = lambda e, lo, hi: f"{e:.3f} ({lo:.3f}–{hi:.3f})"

# ---------- Table 2: sample (development years 2019+2022 and 2025), survey-weighted ----------
d = pd.read_parquet("data/proc/nhis_analytic.parquet")
rows = []
for g in ["No disability", "Any disability"] + DOM:
    def sel(t):
        if g == "No disability": return t[t.disab == 0]
        if g == "Any disability": return t[t.disab == 1]
        return t[t[f"d_{g}"] == 1]
    tr, te = sel(d[d.split == "train"]), sel(d[d.split == "test"])
    wavg = lambda t, c: 100 * np.average(t[c].dropna(), weights=t.loc[t[c].notna(), "wt_pooled"])
    rows.append([LAB[g], f"{len(tr):,}", f1(np.average(tr.age.dropna(), weights=tr.loc[tr.age.notna(), "wt_pooled"])),
                 f1(wavg(tr, "female")), f1(wavg(tr, "phq_pos")), f1(wavg(tr, "gad_pos")), f1(wavg(tr, "depmed")),
                 f1(wavg(tr, "therapy")), f"{len(te):,}", f1(wavg(te, "phq_pos"))])
T[2] = md(pd.DataFrame(rows, columns=["Group", "n (2019+2022)", "Mean age", "Female (%)", "PHQ-8 >= 10 (%)",
                                      "GAD-7 >= 10 (%)", "Depression medication (%)", "Counselling (%)", "n (2025)",
                                      "PHQ-8 >= 10 (%) 2025"]))

# ---------- Table 3: invariance (NHIS 2019+2022) ----------
f = pd.read_csv(O + "invariance_fit.csv")
rows = []
for sc, name in [("phq", "PHQ-8"), ("gad", "GAD-7")]:
    for dm in DOM:
        g = f[(f.scale == sc) & (f.domain == dm)].set_index("step")
        if not {"configural", "scalar"} <= set(g.index): continue
        c, s = g.loc["configural"], g.loc["scalar"]
        rows.append([name, LAB[dm], f"{int(c.n_domain):,}", f3(c["cfi.scaled"]), f3(s["cfi.scaled"]),
                     f"{s['cfi.scaled'] - c['cfi.scaled']:+.3f}", f3(g.loc["strict", "cfi.scaled"]) if "strict" in g.index else "—",
                     f3(c["rmsea.scaled"]), f3(s["rmsea.scaled"])])
T[3] = md(pd.DataFrame(rows, columns=["Scale", "Domain", "n (domain)", "CFI configural", "CFI scalar", "Delta CFI",
                                      "CFI strict", "RMSEA configural", "RMSEA scalar"]))

# ---------- Table 4: classification accuracy of PHQ-8 >= 10 ----------
L = pd.read_csv(O + "corrected_lca_intervals.csv")
S = pd.read_csv(O + "corrected_model_simulations.csv")
sim = S[(S.survey == "NHIS") & (S.scale == "phq") & (S.model == "scalar")]
rows = []
for g in ["No disability", "Any disability"] + DOM:
    l = L[(L.survey == "NHIS") & (L.scale == "dep") & (L.group == g)].iloc[0]
    nh = L[(L.survey == "NHANES") & (L.scale == "dep") & (L.group == g)]
    if g == "No disability":
        m = sim[sim.group == "nodisab"].iloc[0]
    elif g == "Any disability":
        m = None
    else:
        m = sim[(sim.domain == g) & (sim.group == "domain")].iloc[0]
    nh_txt = "—" if nh.empty else \
        f"{100 * (1 - nh.iloc[0].spec):.1f} ({100 * (1 - nh.iloc[0].spec_hi):.1f}–{100 * (1 - nh.iloc[0].spec_lo):.1f})"
    rows.append([LAB[g], "—" if m is None else f1(100 * m.sensitivity), "—" if m is None else f1(100 * m.FPR),
                 f1(100 * l.sens), f"{100 * (1 - l.spec):.1f} ({100 * (1 - l.spec_hi):.1f}–{100 * (1 - l.spec_lo):.1f})", nh_txt])
T[4] = md(pd.DataFrame(rows, columns=["Group", "Sensitivity, model-implied (%)", "FPR, model-implied (%)",
                                      "Sensitivity, latent class (%)", "FPR, latent class (%) (95% CI)",
                                      "FPR, NHANES latent class (%) (95% CI)"]))

# ---------- Tables 5 and 7: full-resource benchmark with CIs ----------
P = pd.read_csv(O + "corrected_performance_points.csv")
I = pd.read_csv(O + "corrected_performance_intervals.csv")
NAME = {"PHQ-8 >= 10 (standard)": "PHQ-8 >= 10 (standard rule)", "GAD-7 >= 10 (standard)": "GAD-7 >= 10 (standard rule)",
        "PHQ-2 >= 3": "PHQ-2 >= 3", "GAD-2 >= 3": "GAD-2 >= 3",
        "Logistic regression (unaware)": "Logistic regression", "XGBoost (unaware)": "XGBoost",
        "Dang: Reweighing (RW)": "Reweighing (Dang et al., 2024)",
        "Dang: Disparate impact remover": "Disparate impact remover (Dang et al., 2024)",
        "Dang: Calibrated EO post-proc (CPP)": "Calibrated EO post-processing (Dang et al., 2024)",
        "Dang: PSTA group thresholds": "PSTA (Dang et al., 2024)",
        "Shipman: fair XGBoost (dual, TPR parity)": "Fair XGBoost, dual (Shipman & Ushakova, 2026)",
        "Tian: fair stacking ensemble (FCSE)": "Fair stacking ensemble (Tian et al., 2026)",
        "Khan: fairness-regularised model": "Fairness-regularised model (Khan et al., 2026)",
        "A3 unaware score + EO thresholds": "Items-only score + cross-fitted EO cut-offs",
        "DAFS-I": "DAFS-I (proposed)"}


def bench(scale, extra):
    order = ([k for k in NAME if k.startswith(("PHQ", "GAD"))] +
             [k for k in NAME if not k.startswith(("PHQ", "GAD", "A3", "DAFS"))] + extra)
    rows = []
    for m in order:
        arm = "matched" if m.startswith(("PHQ", "GAD")) else "crossfit"
        p = P[(P.scale == scale) & (P.target == "yA") & (P.arm == arm) & (P.method == m)]
        if p.empty: continue
        p = p.iloc[0]
        iv = I[(I.scale == scale) & (I.target == "yA") & (I.arm == arm) & (I.method == m)].set_index("metric")
        c = lambda k: ci3(p[k], iv.loc[k, "lo"], iv.loc[k, "hi"])
        name = NAME[m]
        r = [name, f3(p.AUC), c("BAcc"), c("EOD"), c("FPRgap"), c("AOD")]
        if m == "DAFS-I" or (scale == "anx" and m.startswith("A3")):
            r = [f"**{x}**" for x in r]
        rows.append(r)
    return md(pd.DataFrame(rows, columns=["Method", "AUC", "Balanced accuracy", "EOD", "FPR gap", "AOD"]))


T[5] = bench("dep", ["DAFS-I"])
T["G"] = bench("anx", ["A3 unaware score + EO thresholds", "DAFS-I"])

# ---------- Table 6: paired differences, DAFS-I minus comparator (full-resource and matched) ----------
C = pd.read_csv(O + "corrected_paired_differences.csv")
rows = []
pub = [k for k in NAME if k.startswith(("Dang", "Shipman", "Tian", "Khan"))]
for arm, lab, chosen in [("crossfit", "Full resources", "DAFS-I"), ("matched", "Matched split", "DAFS-I + EO")]:
    for m in [k for k in NAME if k.startswith(("PHQ",))] + ["Logistic regression (unaware)", "XGBoost (unaware)"] + pub:
        c = C[(C.scale == "dep") & (C.target == "yA") & (C.arm == arm) & (C.method == chosen) & (C.comparator == m)].set_index("metric")
        if c.empty: continue
        g = lambda k: f"{c.loc[k, 'difference']:+.3f} ({c.loc[k, 'lo']:+.3f} to {c.loc[k, 'hi']:+.3f})"
        rows.append([lab, NAME[m], g("AOD"), g("EOD"), g("FPRgap"), g("BAcc")])
T[6] = md(pd.DataFrame(rows, columns=["Benchmark", "Comparator", "Delta AOD (95% CI)", "Delta EOD (95% CI)",
                                      "Delta FPR gap (95% CI)", "Delta balanced accuracy (95% CI)"]))

# ---------- Table 8: measurement scenarios (replaces the causal decomposition) ----------
rows = []
for dm in DOM:
    sc = S[(S.survey == "NHIS") & (S.scale == "phq") & (S.domain == dm)]
    gd = lambda mdl, grp: sc[(sc.model == mdl) & (sc.group == grp)].iloc[0].FPR * 100
    scal, ref, common = gd("scalar", "domain"), gd("scalar", "nodisab"), gd("scalar_common_reference", "domain")
    lca = L[(L.survey == "NHIS") & (L.scale == "dep") & (L.group == dm)].iloc[0]
    lref = L[(L.survey == "NHIS") & (L.scale == "dep") & (L.group == "No disability")].iloc[0]
    rows.append([LAB[dm], f1(100 * ((1 - lca.spec) - (1 - lref.spec))), f1(scal), f1(common), f1(ref),
                 f"{100 * (scal - common) / (scal - ref):.0f}"])
T[7] = md(pd.DataFrame(rows, columns=["Domain", "Observed gap, latent class (points)", "Model-implied FPR, scalar model (%)",
                                      "Model-implied FPR, reference measurement (%)", "Model-implied FPR, no disability (%)",
                                      "Share of model-implied gap from measurement (%)"]))

# ---------- Table 9: ablation, full resources ----------
rows = []
ABL = [("A1 unaware score + single threshold", "Items-only score (XGBoost), single cut-off"),
       ("A2 domain-aware score + single threshold", "Domain-aware score (XGBoost), single cut-off"),
       ("A3 unaware score + EO thresholds", "Items-only score (XGBoost) + domain EO cut-offs"),
       ("DAFS (proposed)", "Domain-aware score (XGBoost) + domain EO cut-offs"),
       ("DAFS-I", "DAFS-I: item-informed score + domain EO cut-offs (proposed)")]
for scale, lab in [("dep", "PHQ-8"), ("anx", "GAD-7")]:
    for k, name in ABL:
        p = P[(P.scale == scale) & (P.target == "yA") & (P.arm == "crossfit") & (P.method == k)].iloc[0]
        rows.append([lab, name] + [f3(p[c]) for c in ["AUC", "BAcc", "F1", "EOD", "FPRgap", "AOD"]])
T[8] = md(pd.DataFrame(rows, columns=["Scale", "Configuration", "AUC", "Balanced accuracy", "F1", "EOD", "FPR gap", "AOD"]))

# ---------- Table 10: clinician scoring sheet ----------
pts = pd.read_csv(O + "matched_integer_points_yA.csv").set_index("component").points
NM = {"phq81": "1. Little interest or pleasure", "phq82": "2. Feeling down, depressed or hopeless", "phq83": "3. Sleep problems",
      "phq84": "4. Feeling tired or little energy", "phq85": "5. Poor appetite or overeating", "phq86": "6. Feeling bad about yourself",
      "phq87": "7. Trouble concentrating", "phq88": "8. Moving or speaking slowly / restless",
      "tired_x_disab": "Adjustment, item 4, any disability", "conc_x_disab": "Adjustment, item 7, any disability"}
T[9] = md(pd.DataFrame({"Component": [NM[k] for k in pts.index], "Points per response level (0–3)": [int(v) for v in pts.values]}))
thr = [x for x in json.load(open(O + "matched_thresholds_yA.json")) if x["method"] == "integer + EO"][0]["thresholds"]
T["9b"] = md(pd.DataFrame({"Group": [LAB[g] for g in ["none"] + DOM],
                           "Screen positive at (points)": [f">= {int(round(thr[g]))}" for g in ["none"] + DOM]}))

# ---------- Table 11: robustness ----------
r = pd.concat([pd.read_csv(O + "robustness_fpr.csv"), pd.read_csv(O + "robustness_sex_attrition.csv")])
p = r.pivot_table(index="group", columns="check", values="FPR").round(1)
p = p.reindex(["none", "none (no difficulty at all)"] + DOM)
p.index = [LAB.get(i, "No difficulty in any domain") for i in p.index]
p = p.fillna("—").reset_index().rename(columns={
    "index": "Group", "R0 main (2019+2022)": "Main (2019+2022)", "R1 current-only indicators": "Current-only indicators",
    "R2 single-domain only": "Single domain only", "R3 'some difficulty' only": "'Some difficulty' only",
    "R4 test year 2025": "Test year 2025", "R5 women": "Women", "R6 men": "Men",
    "R7 non-response weighted": "Non-response weighted", "R8 equal access (insured, usual care)": "Insured, usual place of care"})
T[10] = md(p)

# ---------- Table 12: three-way screening (DAFS-I, alpha = 0.10, cross-fitted calibration) ----------
R = pd.read_csv(O + "corrected_referral_coverage.csv")
R = R[(R.scale == "dep") & (R.calibration == "OOF")]
rows = []
for g in ["none"] + DOM:
    v = lambda grp, met: R[(R.grouping == grp) & (R.group == g) & (R.metric == met)].iloc[0]
    cm, cd = v("marginal", "coverage"), v("priority-group", "coverage")
    rows.append([LAB[g], f"{100 * cm.estimate:.1f} ({100 * cm.lo:.1f}–{100 * cm.hi:.1f})",
                 f"{100 * cd.estimate:.1f} ({100 * cd.lo:.1f}–{100 * cd.hi:.1f})",
                 f1(100 * v("priority-group", "referral_rate").estimate), f1(100 * v("priority-group", "accuracy_decided").estimate)])
T[11] = md(pd.DataFrame(rows, columns=["Group", "Coverage, single calibration (%) (95% CI)",
                                       "Coverage, domain-conditional (%) (95% CI)", "Refer rate, domain-conditional (%)",
                                       "Accuracy among decided, domain-conditional (%)"]))

json.dump({str(k): v for k, v in T.items()}, open(O + "paper_tables.json", "w"), ensure_ascii=False)
NO = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "G": "7", "7": "8", "8": "9", "9": "10",
      "9b": "10 (continued)", "10": "11", "11": "12"}
with open(O + "paper_tables.md", "w") as fh:
    fh.write("# Tables of the paper (generated by code/30_paper_tables.py)\n\n")
    for k, no in NO.items():
        fh.write(f"## Table {no}\n\n{T[k] if k in T else T[int(k)]}\n\n")
print("written out/paper_tables.json and out/paper_tables.md")
