from pathlib import Path
import sys,os,json,ast,copy
import numpy as np,pandas as pd
r=Path(__file__).resolve().parents[1];os.chdir(r);sys.path.insert(0,'code');from statutils import lca2,bootstrap_weights
src=Path('code/06_latent_class.py').read_text(encoding='utf-8').split('tr = df[df.split')[0];ns={};exec(compile(src,'inspected_LCA_definitions','exec'),ns);df=ns['df'];fit_group=ns['fit_group'];DOMS=ns['DOMS'];tr=df[df.split=='train'].copy();te=df[df.split=='test'].copy();B=200

for scale,ext,items in [('dep',['wg_dep','depmed','depev','therapy'],[f'phq8{i}' for i in range(1,9)]),('anx',['wg_anx','anxmed','anxev','therapy'],[f'gad7{i}' for i in range(1,8)])]:
 fit=tr.dropna(subset=ext);pi,p,meta=lca2(fit[ext].to_numpy(),fit.wt_pooled.to_numpy(),diagnostics=True);d=df.copy();ok=d[ext].notna().all(axis=1);X=d.loc[ok,ext].to_numpy();l0=np.prod(np.where(X==1,p[0],1-p[0]),axis=1)*(1-pi);l1=np.prod(np.where(X==1,p[1],1-p[1]),axis=1)*pi;d['yA']=np.nan;d.loc[ok,'yA']=(l1/(l0+l1)>.5).astype(float);d['yB']=d.phq_pos if scale=='dep' else d.gad_pos
 d['fgroup']='none'
 for k in reversed(['cognition','selfcare','communication','mobility','vision','hearing']):d.loc[d['d_'+k]==1,'fgroup']=k
 d.loc[d.disab.isna(),'fgroup']=np.nan;d['record_id']=d.year.astype(str)+'_'+d.index.astype(str);d=d.dropna(subset=items+['yA','fgroup']);d.to_parquet('data/proc/ml_'+scale+'.parquet');print('ML',scale,d.groupby('split').size().to_dict(),flush=True)
 patterns=np.array([[int(i>>j&1) for j in range(4)] for i in range(16)]);a=np.prod(np.where(patterns==1,p[0],1-p[0]),axis=1)*(1-pi);b=np.prod(np.where(patterns==1,p[1],1-p[1]),axis=1)*pi;pp=b/(a+b);tab=pd.DataFrame(patterns,columns=ext);tab['posterior']=pp;tab['label']=pp>.5;tab.to_csv('out/label_patterns_'+scale+'.csv',index=False)
