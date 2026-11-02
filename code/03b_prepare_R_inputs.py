"""Exports the analysis files read by the R measurement models (04_invariance.R, 13_nhanes_invariance.R)
and harmonises NHANES variable names with the NHIS pipeline. Run after 01_build_nhis.py and 03_nhanes_build.py."""
import pandas as pd
nhis = pd.read_parquet("data/proc/nhis_analytic.parquet")
nhis.to_csv("data/proc/nhis_analytic.csv", index=False)
d = pd.read_parquet("data/proc/nhanes_analytic.parquet")
for i in range(1, 9):
    d[f"phq8{i}"] = d[f"DPQ0{i}0"]
# Washington Group depression item: daily/weekly AND intensity 'a lot' or 'in between'; 'never' -> 0
d["wg_dep"] = (d.FNQ530.isin([1, 2]) & d.FNQ540.isin([2, 3])).astype(float).where(d.FNQ530.between(1, 5))
d.loc[d.FNQ530 == 5, "wg_dep"] = 0.0
d.loc[d.FNQ530.isin([1,2]) & ~d.FNQ540.between(1,3), "wg_dep"] = float("nan")
d["wt_pooled"] = d.WTMEC2YR
d["psu"] = d.SDMVSTRA.astype(int).astype(str) + "_" + d.SDMVPSU.astype(int).astype(str)
d["stratum"] = d.SDMVSTRA
d["split"] = "train"
d = d[d.phq8.notna() & d.disab.notna()]
d.to_csv("data/proc/nhanes_analytic.csv", index=False)
print("NHIS", nhis.shape, "NHANES", d.shape)
