from pathlib import Path
import os,sys,ast,pickle,json
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
r=Path(__file__).resolve().parents[1];os.chdir(r);sys.path.insert(0,'code');from fairmetrics import evaluate,DOMS
YCOL=sys.argv[1] if len(sys.argv)>1 else 'yA';SCALE=os.environ.get('SCALE','dep');SUF='' if SCALE=='dep' else '_anx'
src=Path('code/08_baselines.py').read_text(encoding='utf-8').split('# ---- screening rules ----')[0];ns={};exec(compile(src,'matched_data_split','exec'),ns)
tr,va,te=ns['tr'].reset_index(drop=True),ns['va'].reset_index(drop=True),ns['te'].reset_index(drop=True);ITEMS=ns['ITEMS'];FLAGS=['d_'+k for k in DOMS];GROUPS=np.unique(tr.fgroup);y=va[YCOL].to_numpy();w=va.wt_pooled.to_numpy()
# Import only the inspected threshold functions; no duplicated model/data execution.
tree=ast.parse(Path('code/09_proposed.py').read_text(encoding='utf-8'));env={'np':np,'GROUPS':GROUPS}
from statutils import roc_pts
env['roc_pts']=roc_pts
exec(compile(ast.Module(body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['single_thr','eo_thresholds']],type_ignores=[]),'corrected_threshold_functions','exec'),env)
PRED={};rows=[];details=[]
for config in ['items_xgb','domain_xgb','DAFS-I','integer']:
 if config=='integer' and SCALE=='anx':continue
 def X(t):
  if config=='items_xgb':return t[ITEMS].to_numpy(float)
  if config=='domain_xgb':return t[ITEMS+FLAGS].to_numpy(float)
  z=t[ITEMS].copy()
  if config=='integer':
   z['tired_x_disab']=t.phq84*(t.disab==1);z['conc_x_disab']=t.phq87*(t.disab==1)
   for k in DOMS:z['d_'+k]=t['d_'+k]
  else:
   biased=['phq84','phq87'] if SCALE=='dep' else ['gad71']
   for k in DOMS:
    z['d_'+k]=t['d_'+k]
    for it in biased:z[it+'x'+k]=t[it]*t['d_'+k]
  return z.to_numpy(float)
 xt,xv,xe=X(tr),X(va),X(te);wn=tr.wt_pooled.to_numpy().copy();wn/=wn.mean()
 if config.endswith('xgb'):
  mono='('+','.join(['1']*len(ITEMS)+(['0']*len(FLAGS) if config=='domain_xgb' else []))+')'
  m=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,n_jobs=2,verbosity=0,random_state=0,monotone_constraints=mono).fit(xt,tr[YCOL],sample_weight=wn);sv=m.predict_proba(xv)[:,1];se=m.predict_proba(xe)[:,1]
 else:
  m=LogisticRegression(C=1e6 if config=='integer' else 1,max_iter=5000).fit(xt,tr[YCOL],sample_weight=wn)
  if config=='integer':
   pts=np.round(m.coef_[0][:10]/(m.coef_[0][1]/6)).astype(int);sv=xv[:,:10]@pts;se=xe[:,:10]@pts;pd.DataFrame({'component':ITEMS+['tired_x_disab','conc_x_disab'],'points':pts}).to_csv('out/matched_integer_points_'+YCOL+'.csv',index=False)
  else:sv=m.predict_proba(xv)[:,1];se=m.predict_proba(xe)[:,1]
 for fair in [False,True]:
  name=config+(' + EO' if fair else ' + single')
  if fair:
   thr=env['eo_thresholds'](sv,y,w,va.fgroup.to_numpy(),.05)
   if thr is None:raise RuntimeError('No feasible ROC grid rule '+name)
   q=np.array([se[i]>=thr[g] for i,g in enumerate(te.fgroup)]).astype(int);qv=np.array([sv[i]>=thr[g] for i,g in enumerate(va.fgroup)]).astype(int)
  else:
   t=env['single_thr'](sv,y,w);thr={'all':float(t)};q=(se>=t).astype(int);qv=(sv>=t).astype(int)
  met=evaluate(te,se,q,YCOL);vmet=evaluate(va,sv,qv,YCOL);rows.append(dict(method=name,**met));PRED[name]=(se.astype(float),q.astype(float));details.append(dict(method=name,thresholds={k:float(v) for k,v in thr.items()},actual_calibration_AOD_overlapping=vmet['AOD']))
  print(SCALE,YCOL,name,'AOD',round(met['AOD'],4),'BAcc',round(met['BAcc'],4),flush=True)
pd.DataFrame(rows).to_csv(f'out/matched_proposed_{YCOL}{SUF}.csv',index=False);pickle.dump(PRED,open(f'out/pred_matched_{YCOL}{SUF}.pkl','wb'));Path(f'out/matched_thresholds_{YCOL}{SUF}.json').write_text(json.dumps(details,indent=2));print('MATCHED COMPLETE',SCALE,YCOL,flush=True)

