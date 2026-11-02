# Fairness-Aware Machine Learning for Equitable Depression and Anxiety Screening Across Disability Domains

Code that reproduces every result, table and figure of:

> Jeribi F, Alam S. *Fairness-Aware Machine Learning for Equitable Depression and Anxiety Screening Across Disability Domains.*
> Journal of Disability Research (submitted, 2026).

Corresponding author: **Fathe Jeribi**, Department of Computer Sciences, College of Engineering and Computer Sciences,
and Engineering and Technology Research Center, Jazan University, Jazan, Saudi Arabia (fjeribi@jazanu.edu.sa).

The study audits the PHQ-8 and GAD-7 screening cut-offs (≥ 10) across the six Washington Group disability
domains (vision, hearing, mobility, cognition, self-care, communication), using the US National Health
Interview Survey (NHIS) 2019, 2022 and 2025, with replication in NHANES 2021–2023. It then corrects the
inequity with **DAFS-I**, a disability-aware fair screening method.

## What the code does

| Stage | Question | Methods |
|---|---|---|
| 1. Audit | Do the questionnaires measure the same construct, and does the cut-off classify equally well, in every domain? | Survey-weighted multigroup ordinal CFA (lavaan, WLSMV; configural → strict); model-implied classification accuracy (400,000 simulated respondents per group); two-class latent-class reference standard from indicators external to the questionnaire (multi-start EM, 200 Rao–Wu bootstrap replicates, joint national-count intervals) |
| 2. Explain | Where does the false-positive gap come from? | Measurement scenarios (scalar, strict, reference measurement parameters); equal-access-to-care audit; item differential prediction with design-based sandwich errors and Benjamini–Hochberg FDR |
| 3. Correct | Can a screening rule be accurate and fair in every domain? | **DAFS-I**: survey-weighted L2 logistic score on items, domain flags and pre-selected biased-item × domain interactions, with cross-fitted (PSU-grouped, 5-fold) equalized-odds cut-offs per domain (development target ε = 0.05; tie-aware ROC) |
| Benchmark | Is it better than published methods? | Re-implementation on the same data and label of Dang et al. (2024) reweighing, disparate-impact remover, calibrated EO post-processing and PSTA; Shipman & Ushakova (fairlearn reduction); Tian et al. (fair stacking); Khan et al. (fairness-regularised model). Two equal-resource benchmarks: **full-resource** (primary; all methods on all development data, held-out quantities from the same out-of-fold predictions) and **matched** (all methods on the same 75% of PSUs). Operating-point frontier analysis |
| Deploy | What can services use? | 45-point clinician scoring sheet with domain cut-offs; three-way screening (positive / negative / refer) with domain-conditional calibration and empirical coverage |
| Inference | How certain are the differences? | 500 Rao–Wu PSU bootstrap replicates shared by all methods; paired differences with 95% percentile CIs (no p-values) |

Training uses NHIS 2019 + 2022; NHIS 2025 is the held-out test year.

## Main results (NHIS 2025 test year, full-resource benchmark)

| | AUC | Balanced accuracy | EOD | FPR gap | AOD |
|---|---|---|---|---|---|
| PHQ-8 ≥ 10 (standard cut-off) | 0.831 | 0.643 | 0.524 | 0.243 | 0.368 |
| Fairest published method (PSTA, Dang et al. 2024) | 0.841 | 0.761 | 0.097 | 0.097 | 0.094 |
| **DAFS-I (proposed)** | **0.842** | **0.762** | **0.086** | **0.080** | **0.056** |

Among methods at least as accurate as the standard cut-off, DAFS-I has the lowest AOD, EOD and FPR gap.
Paired 95% CIs exclude zero against reweighing, disparate-impact repair, calibrated post-processing, fair
gradient boosting and fairness regularisation; against PSTA the AOD difference is −0.038 (−0.089 to +0.020) at
equal balanced accuracy, and −0.079 (−0.119 to −0.009) in the matched benchmark. For the GAD-7, the cross-fitted
cut-offs reach AOD 0.068 and balanced accuracy 0.782 (PSTA: 0.069 and 0.774; accuracy difference +0.008,
95% CI +0.005 to +0.011). Under the standard PHQ-8 cut-off, the false-positive rate against the latent-class
reference is 1.1% without disability and 26.1% (95% CI 21.3–31.0) with cognitive disability.
Full results are in `results/`.

## Repository layout

```
├── run_all.sh               one command to reproduce everything
├── requirements.txt         Python packages (exact versions)
├── install_R_packages.R     R packages (lavaan 0.6-17, semTools 0.5-6, jsonlite)
├── data/
│   ├── README.md            sources, file layout, variables used, terms of use
│   └── SHA256SUMS.txt       checksums of the files analysed (checked before every run)
├── code/                    analysis pipeline; code/run_experiments.py runs it in order
└── results/                 outputs of the published run
    ├── tables/              result CSVs and paper_tables.md (all tables, paper numbering)
    ├── figures/             Figure1–Figure7 (300 dpi)
    └── logs/                run logs
```

## How to reproduce

Requirements: Python 3.13 and R 4.3. A full run takes about 2 hours on a 2-core machine (most of it in the
latent-class and comparator bootstraps) and needs about 4 GB of memory.

```bash
git clone https://github.com/<account>/disability-fair-screening.git
cd disability-fair-screening
python3 -m pip install -r requirements.txt
Rscript install_R_packages.R
python3 code/download_data.py      # or place the files manually, see data/README.md
bash run_all.sh                    # writes all outputs to out/
```

All random processes use fixed seeds. Logistic models (including DAFS-I) reproduce exactly; gradient-boosted
models (XGBoost) can differ slightly between operating systems, which affects only the ablation rows that use them.

## Scripts and where their output appears in the paper

| Script | Purpose | Paper |
|---|---|---|
| `run_experiments.py` | Runs every stage in order after verifying the raw-file checksums | — |
| `download_data.py` | Downloads the seven public NCHS files and verifies their checksums | — |
| `01_build_nhis.py`, `03_nhanes_build.py`, `03b_prepare_R_inputs.py` | Harmonised NHIS and NHANES files (all six WG answers required; unknown intensity coded missing; NHANES examination weights) | — |
| `test_corrections.py` | Unit checks of the numerical corrections | — |
| `02_describe.py` | Sample sizes and weighted rates by domain | Table 2 |
| `04_invariance.R`, `13_nhanes_invariance.R` | Multigroup ordinal CFA, NHIS and NHANES | Table 3, Results |
| `corrected_model_simulations.py` | Model-implied accuracy under scalar, strict and reference-measurement scenarios | Tables 4, 8; Figures 3, 5 |
| `corrected_audit.py`, `check_lca_patterns.py`, `06_latent_class.py` | Latent-class reference standard, bootstrap intervals, national counts | Table 4, Figure 3 |
| `cohort_flow.py`, `28_flowchart.py` | Participant flow | Figure 2, Methods |
| `corrected_labels.py` | Supervised labels and fairness groups | Methods |
| `corrected_item_interactions.py`, `25_anx_item_dp.py` | Item differential prediction; biased-item selection | Figure 6 |
| `08_baselines.py`, `matched_proposed.py`, `run_matched_experiments.py` | Matched benchmark (all methods on the same 75% of PSUs) | Table 6, scoring sheet |
| `fullresource_baselines.py` | Full-resource benchmark of the published methods | Tables 5–7, Figure 4 |
| `09_proposed.py`, `18_dafs_variants.py`, `run_corrected_primary.py` | DAFS-I, domain-aware variant and ablations | Tables 5, 7, 9 |
| `corrected_uncertainty.py` | Paired Rao–Wu bootstrap intervals | Tables 5–7 |
| `corrected_referral.py` | Three-way screening coverage | Table 12, Figure 7 |
| `37_lca_sensitivity.py` | Alternative latent-class indicator specifications | Results (robustness) |
| `17_robustness.py`, `27_sex_attrition_power.py` | Robustness checks, sex, non-response weights, insured adults with a usual place of care, power | Table 11, Methods |
| `21_dafsI_eps.py`, `32_calibration.py` | Fairness tolerance and calibration | Results |
| `33_frontier.py`, `34_frontier_table.py` | Fairness across operating points at matched accuracy (`BAND=0.005`) | Table 13 |
| `36_xgb_seeds.py` | Seed stability (10 seeds) of the gradient-boosted variants | Results (ablation) |
| `30_paper_tables.py`, `22_paper_tables.py`, `31_paper_figures.py` | All tables and figures in paper numbering | Tables 1–13, Figures 1, 3–7 |
| `fairmetrics.py`, `statutils.py`, `reference_eo_function.py` | Metrics, Rao–Wu bootstrap, tie-aware ROC, multi-start latent class model | — |

## Using DAFS-I on new data

DAFS-I needs the item responses, the Washington Group domain flags and a reference label for fitting.
`code/18_dafs_variants.py` shows the full procedure: fit the score, obtain cross-fitted out-of-fold scores,
and choose per-domain cut-offs on the equalized-odds grid. For services that cannot run a model, the
scoring sheet in `results/tables/` (`matched_integer_points_yA.csv`, `matched_thresholds_yA.json`) turns DAFS-I
into integer points scored on paper (maximum 45 points; NHIS 2025: AUC 0.840, AOD 0.115, against 0.368 for the
standard cut-off).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). Questions about the code or the paper: Fathe Jeribi (fjeribi@jazanu.edu.sa).

## Funding

The authors extend their appreciation to the King Salman Center for Disability Research
(funder ID: https://doi.org/10.13039/501100019345) for funding this work through Research Group Number KSRG-2026-859.

## License

Code: MIT (see `LICENSE`). Data: NCHS public-use files, subject to the NCHS data-use restrictions
(see `data/README.md`).
