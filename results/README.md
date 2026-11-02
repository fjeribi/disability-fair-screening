# Results of the published run

- `tables/paper_tables.md` — every table of the paper (Tables 1–13) in paper numbering, rendered from the CSV files.
- `tables/*.csv`, `tables/*.json` — all result files written by the pipeline (names as in `out/`).
  Key files: `corrected_lca_intervals.csv` (audit, national counts), `corrected_model_simulations.csv` (model-implied accuracy),
  `fullres_baselines_yA*.csv` and `corrected_performance_*.csv` (benchmarks), `corrected_paired_differences.csv` (paired CIs),
  `corrected_item_interactions.csv` (item differential prediction), `corrected_referral_coverage.csv` (three-way screening),
  `robustness_*.csv`, `power_mdd.csv`, `dafsI_epsilon.csv`, `dafsI_calibration.csv`, `frontier_*.csv`.
- `figures/` — Figures 1–7 (300 dpi PNG) in paper numbering.
- `logs/` — run logs and R session information.
