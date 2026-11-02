# Data

All data are public-use files released by the US National Center for Health Statistics (NCHS).
They are not redistributed in this repository. Run `python code/download_data.py`, or download them
manually and place them as shown below.

## Expected layout

```
data/raw/
├── nhis19/adult19.csv      NHIS 2019 Sample Adult (CSV)
├── nhis22/adult22.csv      NHIS 2022 Sample Adult (CSV)
├── nhis25/adult25.csv      NHIS 2025 Sample Adult (CSV)
├── DEMO_L.xpt              NHANES Aug 2021–Aug 2023, Demographics
├── DPQ_L.xpt               NHANES, Mental Health – Depression Screener (PHQ-9)
├── FNQ_L.xpt               NHANES, Functioning (Washington Group questions)
└── HUQ_L.xpt               NHANES, Hospital Utilization & Access to Care
```

`data/SHA256SUMS.txt` lists the checksums of the exact files analysed (downloaded October 2026).
`python code/run_experiments.py` checks them before it starts.

## Sources

- **NHIS:** https://www.cdc.gov/nchs/nhis/documentation/index.html
  (choose the year → Sample Adult → CSV data file)
- **NHANES 2021–2023:** https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Questionnaire&Cycle=2021-2023
  (DEMO_L is under Component=Demographics)

## Variables used

| Construct | NHIS | NHANES |
|---|---|---|
| PHQ-8 items | `PHQ81_A` … `PHQ88_A` | `DPQ010` … `DPQ080` |
| GAD-7 items | `GAD71_A` … `GAD77_A` | not available |
| Washington Group domains | `VISIONDF_A`, `HEARINGDF_A`, `DIFF_A`, `COGMEMDFF_A`, `UPPSLFCR_A`, `COMDIFF_A` | `FNQ410`, `FNQ430`, `FNQ440`, `FNQ450`, `FNQ460`, `FNQ470` |
| Indicators for the latent-class reference standard (besides the screener result) | `DEPMED_A`, `ANXMED_A`, `MHTHRPY_A`, `DEPEV_A`, `ANXEV_A`, `DEPFREQ_A`, `DEPLEVEL_A`, `ANXFREQ_A`, `ANXLEVEL_A` | `HUQ090` (counselling), `FNQ530`/`FNQ540` (Washington Group affect item) |
| Covariates | `AGEP_A`, `SEX_A`, `RATCAT_A`, `HISPALLP_A`, `EDUC_A`/`EDUCP_A`, `NOTCOV_A`, `USUALPL_A` | `RIDAGEYR`, `RIAGENDR` |
| Survey design | `WTFA_A`, `PSTRAT`, `PPSU` | `WTMEC2YR`, `SDMVSTRA`, `SDMVPSU` |

A domain disability is "a lot of difficulty" or "cannot do at all" on that Washington Group item.
The 2019 and 2022 files are pooled for training (weights divided by 2); 2025 is the held-out test year.

## Terms of use

NCHS public-use data may be used only for statistical reporting and analysis. Users must not attempt
to identify any individual respondent. See https://www.cdc.gov/nchs/data_access/restrictions.htm

