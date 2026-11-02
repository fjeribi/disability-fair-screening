"""Manuscript figures from the corrected outputs (300 dpi PNG, colour-vision-safe palette)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
O = "out/"
DOM = ["cognition", "selfcare", "communication", "mobility", "vision", "hearing"]
LAB = {"cognition": "Cognition", "selfcare": "Self-care", "communication": "Communication", "mobility": "Mobility",
       "vision": "Vision", "hearing": "Hearing", "none": "No disability", "No disability": "No disability"}

# ---------- Figure 1: study framework ----------
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 50)
def box(x, y, w, h, title, body, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=edge, lw=1.4))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.8, color=MUTED, linespacing=1.35)
box(1, 26, 22, 22, "Data", "NHIS 2019+2022 (train)\nNHIS 2025 (test)\nNHANES 2021-23\n(replication)\nPHQ-8, GAD-7, WG-SS", MUTED)
box(27, 26, 22, 22, "Stage 1: Audit", "Measurement invariance\nper WG domain\n\nModel-implied and\nlatent-class accuracy\nof the ≥ 10 cut-off", BLUE)
box(53, 26, 22, 22, "Stage 2: Explain", "Measurement scenarios;\nitem x domain\ndifferential prediction;\naccess-to-care and\nsex-specific checks", ORANGE)
box(79, 26, 20, 22, "Stage 3: Correct", "DAFS-I: item-informed\nscore; survey-\nweighted, cross-fitted\nequalized-odds\ncut-offs per domain", AQUA)
box(27, 1, 34, 19, "Superiority benchmark", "Equal training resources:\nPHQ-8/PHQ-2 rules; Dang et al.\n(RW, DIR, CPP, PSTA); Shipman;\nTian; Khan. AUC, BAcc,\nEOD, FPR gap, AOD", MUTED)
box(65, 1, 34, 19, "Deployment outputs", "Integer clinician scoring sheet\nwith domain cut-offs; three-way\nscreening (positive / negative /\nrefer) with domain coverage", MUTED)
for x0, x1 in [(23.6, 26.4), (49.6, 52.4), (75.6, 78.4)]:
    ax.annotate("", xy=(x1, 37), xytext=(x0, 37), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))
ax.annotate("", xy=(44, 20.6), xytext=(44, 25.4), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
ax.annotate("", xy=(89, 20.6), xytext=(89, 25.4), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))
fig.savefig(O + "fig1_framework.png"); plt.close(fig)

# ---------- Figure 3: false-positive rate by domain, three lines of evidence ----------
L = pd.read_csv(O + "corrected_lca_intervals.csv")
S = pd.read_csv(O + "corrected_model_simulations.csv")
sim = S[(S.survey == "NHIS") & (S.scale == "phq") & (S.model == "scalar")]
lca = L[(L.survey == "NHIS") & (L.scale == "dep")].set_index("group")
nh = L[(L.survey == "NHANES") & (L.scale == "dep")].set_index("group")
rows = ["none"] + DOM
key = lambda r: "No disability" if r == "none" else r
y = np.arange(len(rows))[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
v_l = [100 * (1 - lca.loc[key(r), "spec"]) for r in rows]
lo = [100 * (1 - lca.loc[key(r), "spec_hi"]) for r in rows]
hi = [100 * (1 - lca.loc[key(r), "spec_lo"]) for r in rows]
v_m = [100 * sim[sim.group == "nodisab"].FPR.iloc[0]] + [100 * sim[(sim.domain == r) & (sim.group == "domain")].FPR.iloc[0] for r in DOM]
v_n = [100 * (1 - nh.loc[key(r), "spec"]) if key(r) in nh.index else np.nan for r in rows]
ax.hlines(y + 0.18, lo, hi, color=BLUE, lw=1.6)
ax.scatter(v_l, y + 0.18, s=34, color=BLUE, marker="o", zorder=3, ec="white", lw=1, label="NHIS latent-class reference (95% CI)")
ax.scatter(v_m, y, s=34, color=ORANGE, marker="s", zorder=3, ec="white", lw=1, label="NHIS model-implied (no external labels)")
ax.scatter(v_n, y - 0.18, s=40, color=AQUA, marker="D", zorder=3, ec="white", lw=1, label="NHANES latent-class replication")
ax.set_yticks(y); ax.set_yticklabels([LAB[r] for r in rows]); ax.set_xlabel("False-positive rate of PHQ-8 ≥ 10 (%)")
ax.grid(axis="y", visible=False); ax.set_xlim(0, 33)
ax.legend(loc="lower right", fontsize=7.5)
fig.savefig(O + "fig3_fpr_by_domain.png"); plt.close(fig)

# ---------- Figure 4: accuracy-fairness plane, full-resource benchmark ----------
P = pd.read_csv(O + "corrected_performance_points.csv")
P = P[(P.scale == "dep") & (P.target == "yA")]
rules = P[(P.arm == "matched") & P.method.str.startswith("PHQ")]
fr = P[(P.arm == "crossfit") & P.method.str.contains("Dang|Shipman|Tian|Khan|Logistic|XGBoost \\(unaware\\)", regex=True)]
b = pd.concat([rules, fr])
short = {"PHQ-8 >= 10 (standard)": "PHQ-8≥10", "PHQ-2 >= 3": "PHQ-2≥3", "Logistic regression (unaware)": "LR",
         "XGBoost (unaware)": "XGBoost", "Dang: Reweighing (RW)": "Dang RW", "Dang: Disparate impact remover": "Dang DIR",
         "Dang: Calibrated EO post-proc (CPP)": "Dang CPP", "Dang: PSTA group thresholds": "Dang PSTA",
         "Shipman: fair XGBoost (dual, TPR parity)": "Shipman", "Tian: fair stacking ensemble (FCSE)": "Tian FCSE",
         "Khan: fairness-regularised model": "Khan"}
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.scatter(b.BAcc, b.AOD, s=34, color=BLUE, marker="o", ec="white", lw=1, zorder=3, label="Published rules and methods")
off = {"Dang RW": (4, -9), "XGBoost": (4, 3), "LR": (-14, 4), "Dang CPP": (4, 2), "PHQ-8≥10": (4, 2), "Dang PSTA": (4, 3)}
for _, r in b.iterrows():
    dx, dy = off.get(short[r.method], (4, 2))
    ax.annotate(short[r.method], (r.BAcc, r.AOD), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=MUTED)
p = P[(P.arm == "crossfit") & (P.method == "DAFS-I")].iloc[0]
ax.scatter([p.BAcc], [p.AOD], s=70, color=ORANGE, marker="*", ec="white", lw=0.8, zorder=4, label="DAFS-I (proposed)")
ax.annotate("DAFS-I", (p.BAcc, p.AOD), xytext=(5, -9), textcoords="offset points", fontsize=7.5, color=INK, weight="bold")
ax.set_xlabel("Balanced accuracy (higher is better)"); ax.set_ylabel("Average odds difference (lower is fairer)")
ax.set_ylim(0, 0.6); ax.legend(loc="upper left", fontsize=7.5)
fig.savefig(O + "fig4_accuracy_fairness.png"); plt.close(fig)

# ---------- Figure 5: measurement scenarios ----------
order = DOM[::-1]
sc = S[(S.survey == "NHIS") & (S.scale == "phq")]
g = lambda dm, mdl, grp: 100 * sc[(sc.domain == dm) & (sc.model == mdl) & (sc.group == grp)].FPR.iloc[0]
scal = np.array([g(d, "scalar", "domain") for d in order]); com = np.array([g(d, "scalar_common_reference", "domain") for d in order])
ref = np.array([g(d, "scalar", "nodisab") for d in order])
yy = np.arange(len(order))
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.barh(yy, com - ref, left=ref, height=0.62, color=BLUE, ec="white", lw=1.5, label="Latent-distribution difference")
ax.barh(yy, scal - com, left=com, height=0.62, color=ORANGE, ec="white", lw=1.5, label="Measurement-parameter difference")
ax.barh(yy, ref, height=0.62, color="#d9d8d4", ec="white", lw=1.5, label="No-disability rate")
for i in range(len(order)):
    ax.text(scal[i] + 0.3, i, f"{scal[i]:.1f}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([LAB[d] for d in order]); ax.grid(axis="y", visible=False)
ax.set_xlabel("Model-implied false-positive rate of PHQ-8 ≥ 10 (%)")
ax.legend(loc="lower right", fontsize=7.5)
fig.savefig(O + "fig5_measurement_scenarios.png"); plt.close(fig)

# ---------- Figure 6: item-level differential prediction (BH-adjusted) ----------
it = pd.read_csv(O + "corrected_item_interactions.csv"); it = it[it.scale == "dep"]
items = [f"phq8{i}" for i in range(1, 9)]
names = ["1 interest", "2 depressed", "3 sleep", "4 tired", "5 appetite", "6 failure", "7 concentrating", "8 psychomotor"]
M = it.pivot(index="item", columns="domain", values="OR_ratio").loc[items, DOM]
Q = it.pivot(index="item", columns="domain", values="q_BH").loc[items, DOM]
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
cmap = LinearSegmentedColormap.from_list("div", ["#c94f1d", "#f3c2ab", "#f0efec", "#a9c9f0", "#1c5cab"])
fig, ax = plt.subplots(figsize=(6.0, 3.6))
im = ax.imshow(np.log(M.to_numpy()), cmap=cmap, norm=TwoSlopeNorm(0, -0.55, 0.55), aspect="auto")
for i in range(len(items)):
    for j in range(len(DOM)):
        ax.text(j, i, f"{M.iloc[i, j]:.2f}{'*' if Q.iloc[i, j] < 0.05 else ''}", ha="center", va="center", fontsize=7.5, color=INK)
ax.set_xticks(range(len(DOM))); ax.set_xticklabels([LAB[d] for d in DOM], rotation=30, ha="right")
ax.set_yticks(range(len(items))); ax.set_yticklabels([f"PHQ item {s}" for s in names]); ax.grid(False)
cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_ticks(np.log([0.6, 0.8, 1, 1.25, 1.6])); cb.set_ticklabels(["0.60", "0.80", "1.00", "1.25", "1.60"])
cb.set_label("Ratio of odds ratios (domain / no disability)", fontsize=7.5)
fig.savefig(O + "fig6_item_bias.png"); plt.close(fig)

# ---------- Figure 7: three-way screening coverage ----------
R = pd.read_csv(O + "corrected_referral_coverage.csv")
R = R[(R.scale == "dep") & (R.calibration == "OOF") & (R.metric == "coverage")]
grp = ["none"] + DOM
x = np.arange(len(grp))
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for k, (gr, color, mk, dx) in enumerate([("marginal", BLUE, "o", -0.12), ("priority-group", ORANGE, "s", 0.12)]):
    v = [100 * R[(R.grouping == gr) & (R.group == g_)].estimate.iloc[0] for g_ in grp]
    ax.scatter(x + dx, v, s=36, color=color, marker=mk, ec="white", lw=1, zorder=3,
               label="Single (marginal) calibration" if k == 0 else "Domain-conditional calibration (proposed)")
ax.axhline(90, color=MUTED, lw=1, ls="--"); ax.text(-0.45, 92.4, "target 90%", ha="left", fontsize=7, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([LAB[g_] for g_ in grp], rotation=20, ha="right"); ax.grid(axis="x", visible=False)
ax.set_ylabel("Coverage on NHIS 2025 (%)"); ax.set_ylim(65, 100); ax.legend(loc="lower left", fontsize=7.5)
fig.savefig(O + "fig7_coverage.png"); plt.close(fig)
print("figures written")
