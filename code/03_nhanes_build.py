"""NHANES 2021-23 replication file (depression only): PHQ-8 from DPQ_L, WG domains from FNQ_L, counselling HUQ090.
The public prescription file (RXQ_RX_L) records no drug names, so no medication indicator is used."""
import numpy as np, pandas as pd
R="data/raw/"
demo=pd.read_sas(R+"DEMO_L.xpt"); dpq=pd.read_sas(R+"DPQ_L.xpt"); fnq=pd.read_sas(R+"FNQ_L.xpt"); huq=pd.read_sas(R+"HUQ_L.xpt")
df=demo[["SEQN","RIDAGEYR","RIAGENDR","WTINT2YR","WTMEC2YR","SDMVSTRA","SDMVPSU"]].merge(dpq,on="SEQN").merge(fnq,on="SEQN",how="left").merge(huq[["SEQN","HUQ090"]],on="SEQN",how="left")
items=[f"DPQ0{i}0" for i in range(1,9)]
for c in items: df[c]=df[c].where(df[c].between(0,3)).round()
P=df[items]
df["phq8"]=np.where(P.isna().sum(axis=1)<=1,(P.mean(axis=1)*8).round(),np.nan)
df["phq_pos"]=(df.phq8>=10).astype(float).where(df.phq8.notna())
dom={"vision":"FNQ410","hearing":"FNQ430","mobility":"FNQ440","communication":"FNQ450","cognition":"FNQ460","selfcare":"FNQ470"}
for k,c in dom.items():
    lvl=df[c].where(df[c].between(1,4))-1
    df["d_"+k]=(lvl>=2).astype(float).where(lvl.notna())
df["disab"]=(df[[f"d_{k}" for k in dom]].sum(axis=1,min_count=len(dom))>=1).astype(float).where(df[[f"d_{k}" for k in dom]].notna().all(axis=1))
df[[f"d_{k}" for k in dom]]=df[[f"d_{k}" for k in dom]].where(df.disab.notna(),np.nan)
df["therapy"]=df.HUQ090.map({1:1.0,2:0.0})
df.to_parquet("data/proc/nhanes_analytic.parquet")
print("n with PHQ-8:",df.phq8.notna().sum(), " with WG items:", df.disab.notna().sum())
for g,m in [("No disability",df.disab==0),("Any disability",df.disab==1)]+[(k,df["d_"+k]==1) for k in dom]:
    t=df[m&df.phq8.notna()]; w=t.WTINT2YR
    print(f"{g:15s} n={len(t):5d} cases={int(t.phq_pos.sum()):4d} PHQ>=10 {100*np.average(t.phq_pos,weights=w):5.1f}%  therapy {100*np.average(t.therapy.fillna(0),weights=w):5.1f}%")

