"""Stage 1: build the harmonised NHIS 2019/2022/2025 analysis file.

Output: data/proc/nhis_analytic.parquet
- PHQ-8 / GAD-7 items rescored 0-3, totals, standard cut-offs (>=10)
- Washington Group domains: 'a lot of difficulty' or 'cannot do' = domain disability (WG cut-off 3)
- Graded domain level 0-3 (none / some / a lot / cannot)
- Indicators for the latent-class reference standard
- Survey design: stratum, PSU (year-prefixed), weights pooled per split
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data/raw", ROOT / "data/proc"
PROC.mkdir(parents=True, exist_ok=True)

PHQ = [f"PHQ8{i}_A" for i in range(1, 9)]
GAD = [f"GAD7{i}_A" for i in range(1, 8)]
DOMAINS = {"vision": "VISIONDF_A", "hearing": "HEARINGDF_A", "mobility": "DIFF_A",
           "cognition": "COGMEMDFF_A", "selfcare": "UPPSLFCR_A", "communication": "COMDIFF_A"}
YESNO = {"depmed": "DEPMED_A", "anxmed": "ANXMED_A", "therapy": "MHTHRPY_A",
         "depev": "DEPEV_A", "anxev": "ANXEV_A", "uninsured": "NOTCOV_A"}
OTHER = ["DEPFREQ_A", "DEPLEVEL_A", "ANXFREQ_A", "ANXLEVEL_A", "USUALPL_A", "AGEP_A", "SEX_A",
         "RATCAT_A", "HISPALLP_A", "WTFA_A", "PSTRAT", "PPSU", "DISAB3_A"]
EDU = {"19": "EDUC_A", "22": "EDUCP_A", "25": "EDUCP_A"}


def yn(s):
    return s.map({1: 1.0, 2: 0.0})


def build_year(y):
    cols = PHQ + GAD + list(DOMAINS.values()) + list(YESNO.values()) + OTHER + [EDU[y]]
    df = pd.read_csv(RAW / f"nhis{y}/adult{y}.csv", usecols=cols, low_memory=False)
    out = pd.DataFrame(index=df.index)
    out["year"] = 2000 + int(y)
    # items: 1..4 -> 0..3 ; 7/8/9 -> NaN
    for c in PHQ + GAD:
        out[c.replace("_A", "").lower()] = df[c].where(df[c].between(1, 4)) - 1
    phq = out[[c.replace("_A", "").lower() for c in PHQ]]
    gad = out[[c.replace("_A", "").lower() for c in GAD]]
    # NCHS prorating rule: allow one missing item (mean-impute), else total missing
    out["phq8"] = np.where(phq.isna().sum(axis=1) <= 1, phq.mean(axis=1) * 8, np.nan)
    out["gad7"] = np.where(gad.isna().sum(axis=1) <= 1, gad.mean(axis=1) * 7, np.nan)
    out["phq8"] = out["phq8"].round()
    out["gad7"] = out["gad7"].round()
    out["phq_pos"] = (out["phq8"] >= 10).astype(float).where(out["phq8"].notna())
    out["gad_pos"] = (out["gad7"] >= 10).astype(float).where(out["gad7"].notna())
    for k, c in DOMAINS.items():
        lvl = df[c].where(df[c].between(1, 4)) - 1          # 0 none,1 some,2 a lot,3 cannot
        out[f"lvl_{k}"] = lvl
        out[f"d_{k}"] = (lvl >= 2).astype(float).where(lvl.notna())
    dcols = [f"d_{k}" for k in DOMAINS]
    out["n_domains"] = out[dcols].sum(axis=1, min_count=len(dcols))
    out["disab"] = (out["n_domains"] >= 1).astype(float).where(out["n_domains"].notna())
    out[dcols] = out[dcols].where(out[dcols].notna().all(axis=1), np.nan)
    for k, c in YESNO.items():
        out[k] = yn(df[c])
    out["usual_place"] = df["USUALPL_A"].map({1: 1.0, 2: 0.0, 3: 1.0})
    # WG affect items (frequency 1=daily..5=never; level 1=a little,2=a lot,3=between)
    out["depfreq"] = df["DEPFREQ_A"].where(df["DEPFREQ_A"].between(1, 5))
    out["anxfreq"] = df["ANXFREQ_A"].where(df["ANXFREQ_A"].between(1, 5))
    out["deplevel"] = df["DEPLEVEL_A"].where(df["DEPLEVEL_A"].between(1, 3))
    out["anxlevel"] = df["ANXLEVEL_A"].where(df["ANXLEVEL_A"].between(1, 3))
    out["age"] = df["AGEP_A"].where(df["AGEP_A"] < 97)
    out["female"] = df["SEX_A"].map({1: 0.0, 2: 1.0})
    out["income_ratio_cat"] = df["RATCAT_A"].where(df["RATCAT_A"] < 98)
    out["race_eth"] = df["HISPALLP_A"].where(df["HISPALLP_A"] < 97)
    out["educ"] = df[EDU[y]].where(df[EDU[y]] < 97)
    out["wt"] = df["WTFA_A"]
    out["stratum"] = df["PSTRAT"]
    out["psu"] = df["PSTRAT"].astype(str) + "_" + df["PPSU"].astype(str)
    out["disab3_nchs"] = df["DISAB3_A"].map({1: 1.0, 2: 0.0})
    return out


df = pd.concat([build_year(y) for y in ["19", "22", "25"]], ignore_index=True)
df["split"] = np.where(df["year"] == 2025, "test", "train")
# pooled-year weights: divide by number of years in the split (NCHS guidance)
df["wt_pooled"] = np.where(df["split"] == "train", df["wt"] / 2, df["wt"])
# design cluster unique within year
df["psu"] = df["year"].astype(str) + "_" + df["psu"]
df["stratum"] = df["year"].astype(str) + "_" + df["stratum"].astype(str)
df.to_parquet(PROC / "nhis_analytic.parquet")
print(df.shape)
# sanity check: our disability flag vs NCHS DISAB3_A
print("agreement with NCHS DISAB3_A:", (df["disab"] == df["disab3_nchs"]).mean().round(4))
