from pathlib import Path
import sys,os,json,ast,copy
import numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];os.chdir(r);sys.path.insert(0,'code');from statutils import lca2,bootstrap_weights
src=Path('code/06_latent_class.py').read_text(encoding='utf-8').split('tr = df[df.split')[0];ns={};exec(compile(src,'inspected_LCA_definitions','exec'),ns);df=ns['df'];fit_group=ns['fit_group'];DOMS=ns['DOMS'];tr=df[df.split=='train'].copy();te=df[df.split=='test'].copy();B=200
# Corrected point estimates; saved multistart and pattern diagnostics.
rows=[];diags=[]
for survey,t,indsets in [('NHIS',tr,ns['IND']),('NHANES',pd.read_csv('data/proc/nhanes_analytic.csv'),{'dep':['phq_pos','wg_dep','therapy']})]:
 groups={'No disability':t.disab==0,'Any disability':t.disab==1};groups.update({k:t['d_'+k]==1 for k in DOMS})
 for scale,ind in indsets.items():
  for group,mask in groups.items():
   tt=t.loc[mask].dropna(subset=ind);pi,p,meta=lca2(tt[ind].to_numpy(),tt.wt_pooled.to_numpy(),diagnostics=True)
   if survey=='NHANES' and len(tt)<120:continue
   pop=t.loc[mask,'wt_pooled'].sum();row=dict(survey=survey,scale=scale,group=group,n=len(tt),latent_prev=pi,sens=p[1,0],spec=1-p[0,0],population=pop,FP_millions=pop*(1-pi)*p[0,0]/1e6)
   rows.append(row);diags.append(dict(survey=survey,scale=scale,group=group,probabilities=p.tolist(),**meta));print('POINT',survey,scale,group,'FPR',round(100*p[0,0],3),flush=True)
pd.DataFrame(rows).to_csv('out/corrected_lca_points.csv',index=False);Path('out/lca_fit_diagnostics.json').write_text(json.dumps(diags,indent=2))
# One common PSU draw per replicate; refit each group, including prevalence and population.
boots=[];rng=np.random.default_rng(991)
for survey,t,indsets in [('NHIS',tr,ns['IND']),('NHANES',pd.read_csv('data/proc/nhanes_analytic.csv'),{'dep':['phq_pos','wg_dep','therapy']})]:
 for b in range(B):
  w=bootstrap_weights(t,rng);bt=t[w>0].copy();bt['wt_pooled']=w[w>0]
  groups={'No disability':bt.disab==0,'Any disability':bt.disab==1};groups.update({k:bt['d_'+k]==1 for k in DOMS})
  for scale,ind in indsets.items():
   for group,mask in groups.items():
    if survey=='NHANES' and not any(x['survey']==survey and x['group']==group for x in rows):continue
    tt=bt.loc[mask].dropna(subset=ind)
    try:
     pi,p=lca2(tt[ind].to_numpy(),tt.wt_pooled.to_numpy(),starts=3);pop=bt.loc[mask,'wt_pooled'].sum();boots.append(dict(survey=survey,scale=scale,group=group,replicate=b,latent_prev=pi,sens=p[1,0],spec=1-p[0,0],population=pop,FP_millions=pop*(1-pi)*p[0,0]/1e6))
    except ValueError as e:raise RuntimeError((survey,group,b,str(e)))
  if (b+1)%20==0:print('LCA BOOT',survey,b+1,'/',B,flush=True);pd.DataFrame(boots).to_csv('out/corrected_lca_bootstrap.csv',index=False)
pd.DataFrame(boots).to_csv('out/corrected_lca_bootstrap.csv',index=False);bs=pd.DataFrame(boots);point=pd.DataFrame(rows)
for metric in ['latent_prev','sens','spec','FP_millions']:
 qs=bs.groupby(['survey','scale','group'])[metric].quantile([.025,.975]).unstack();point=point.merge(qs.rename(columns={.025:metric+'_lo',.975:metric+'_hi'}),on=['survey','scale','group'])
point.to_csv('out/corrected_lca_intervals.csv',index=False)
# Create corrected supervised cohorts independently of 5-indicator audit.
for scale,ext,items in [('dep',['wg_dep','depmed','depev','therapy'],[f'phq8{i}' for i in range(1,9)]),('anx',['wg_anx','anxmed','anxev','therapy'],[f'gad7{i}' for i in range(1,8)])]:
 fit=tr.dropna(subset=ext);pi,p,meta=lca2(fit[ext].to_numpy(),fit.wt_pooled.to_numpy(),diagnostics=True);d=df.copy();ok=d[ext].notna().all(axis=1);X=d.loc[ok,ext].to_numpy();l0=np.prod(np.where(X==1,p[0],1-p[0]),axis=1)*(1-pi);l1=np.prod(np.where(X==1,p[1],1-p[1]),axis=1)*pi;d['yA']=np.nan;d.loc[ok,'yA']=(l1/(l0+l1)>.5).astype(float);d['yB']=d.phq_pos if scale=='dep' else d.gad_pos
 d['fgroup']='none'
 for k in reversed(['cognition','selfcare','communication','mobility','vision','hearing']):d.loc[d['d_'+k]==1,'fgroup']=k
 d.loc[d.disab.isna(),'fgroup']=np.nan;d['record_id']=d.year.astype(str)+'_'+d.index.astype(str);d=d.dropna(subset=items+['yA','fgroup']);d.to_parquet('data/proc/ml_'+scale+'.parquet');print('ML',scale,d.groupby('split').size().to_dict(),flush=True)
 patterns=np.array([[int(i>>j&1) for j in range(4)] for i in range(16)]);a=np.prod(np.where(patterns==1,p[0],1-p[0]),axis=1)*(1-pi);b=np.prod(np.where(patterns==1,p[1],1-p[1]),axis=1)*pi;pp=b/(a+b);tab=pd.DataFrame(patterns,columns=ext);tab['posterior']=pp;tab['label']=pp>.5;tab.to_csv('out/label_patterns_'+scale+'.csv',index=False)
Path('out/audit_complete.json').write_text(json.dumps({'B':B,'bootstrap':'Rao-Wu PSU rescaling within strata; full-data LCA refits per draw','label_uncertainty_in_ML_test_intervals':'not included'},indent=2));print('AUDIT AND LABELS COMPLETE',flush=True)
