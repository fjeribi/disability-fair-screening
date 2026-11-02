import pathlib,os,pandas as pd,numpy as np
r=pathlib.Path(__file__).resolve().parents[1];os.chdir(r);d=pd.read_parquet('data/proc/nhis_analytic.parquet');rows=[]
for scale,stem,n in [('dep','phq8',8),('anx','gad7',7)]:
 m=pd.read_parquet(f'data/proc/ml_{scale}.parquet')
 for year,t in d.groupby('year'):
  v=m[m.year==year];rows.append(dict(year=year,scale=scale,raw_n=len(t),complete_disability=int(t.disab.notna().sum()),complete_items=int(t[[stem+str(i) for i in range(1,n+1)]].notna().all(axis=1).sum()),final_supervised_n=len(v),latent_positive_n=int(v.yA.sum()),questionnaire_positive_n=int(v.yB.sum()),overlapping_domain_n=int((v[['d_'+g for g in ['vision','hearing','mobility','cognition','selfcare','communication']]].sum(axis=1)>1).sum())))
pd.DataFrame(rows).to_csv('out/corrected_cohort_flow.csv',index=False);print('COHORT FLOW COMPLETE')
