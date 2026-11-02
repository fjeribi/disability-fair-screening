"""Conditional paired survey-PSU percentile intervals for fixed fitted screening rules.
Training/LCA/selection uncertainty is not included. No bootstrap sign fractions are called p-values.
"""
import os,sys,pathlib,pickle,numpy as np,pandas as pd,json
r=pathlib.Path(__file__).resolve().parents[1];os.chdir(r);sys.path.insert(0,'code');from statutils import bootstrap_weights
from fairmetrics import DOMS,evaluate
B=500;points=[];intervals=[];contrasts=[];support=[]
for scale in ['dep','anx']:
 d=pd.read_parquet('data/proc/ml_'+scale+'.parquet');te=d[d.split=='test'].reset_index(drop=True);suf='' if scale=='dep' else '_anx';masks=np.column_stack([(te.disab==0).to_numpy()]+[(te['d_'+k]==1).to_numpy() for k in DOMS]).astype(float)
 rng=np.random.default_rng(842);weights=np.array([bootstrap_weights(te,rng) for _ in range(B)])
 for target in ['yA','yB']:
  y=te[target].to_numpy(float)
  for i,g in enumerate(['none']+DOMS):support.append(dict(scale=scale,target=target,group=g,n=int(masks[:,i].sum()),cases=int((masks[:,i]*y).sum()),noncases=int((masks[:,i]*(1-y)).sum())))
  for arm in ['matched','crossfit']:
   if arm=='crossfit' and target=='yB':continue
   if arm=='matched':files=[f'out/pred_baselines_{target}{suf}.pkl',f'out/pred_matched_{target}{suf}.pkl'];chosen='DAFS-I + EO'
   else:files=[f'out/pred_proposed_yA{suf}_eps0.05.pkl',f'out/pred_dafsI{suf}.pkl']+([f'out/pred_fullres_baselines_yA{suf}.pkl'] if pathlib.Path(f'out/pred_fullres_baselines_yA{suf}.pkl').exists() else []);chosen='DAFS-I'
   pred={}
   for file in files:
    if not pathlib.Path(file).exists():raise FileNotFoundError(file)
    pred.update(pickle.load(open(file,'rb')))
   names=list(pred);Q=np.array([np.clip(pred[n][1],0,1) for n in names]);allmask=np.column_stack([np.ones(len(te)),masks]);draws=[]
   for w in weights:
    wp=w[:,None]*y[:,None]*allmask;wn=w[:,None]*(1-y[:,None])*allmask
    tp=Q@wp/wp.sum(axis=0);fp=Q@wn/wn.sum(axis=0)
    dt=np.abs(tp[:,2:]-tp[:,1,None]);df=np.abs(fp[:,2:]-fp[:,1,None]);draws.append(np.column_stack([(tp[:,0]+1-fp[:,0])/2,np.max((dt+df)/2,axis=1),np.max(dt,axis=1),np.max(df,axis=1)]))
   draws=np.array(draws);metrics=['BAcc','AOD','EOD','FPRgap']
   for j,name in enumerate(names):
    met=evaluate(te,np.asarray(pred[name][0]),np.clip(pred[name][1],0,1),target);points.append(dict(scale=scale,target=target,arm=arm,method=name,**met))
    for k,metric in enumerate(metrics):
     z=draws[:,j,k];lo,hi=np.nanquantile(z,[.025,.975]);intervals.append(dict(scale=scale,target=target,arm=arm,method=name,metric=metric,estimate=met[metric],lo=lo,hi=hi,valid_replicates=int(np.isfinite(z).sum()),B=B))
   j=names.index(chosen)
   for h,name in enumerate(names):
    if h==j:continue
    for k,metric in enumerate(metrics):
     diff=draws[:,j,k]-draws[:,h,k];lo,hi=np.nanquantile(diff,[.025,.975]);contrasts.append(dict(scale=scale,target=target,arm=arm,method=chosen,comparator=name,metric=metric,difference=points[-len(names)+j][metric]-points[-len(names)+h][metric],lo=lo,hi=hi,valid_replicates=int(np.isfinite(diff).sum()),B=B))
   print(scale,target,arm,'intervals complete',flush=True)
pd.DataFrame(points).to_csv('out/corrected_performance_points.csv',index=False);pd.DataFrame(intervals).to_csv('out/corrected_performance_intervals.csv',index=False);pd.DataFrame(contrasts).to_csv('out/corrected_paired_differences.csv',index=False);pd.DataFrame(support).to_csv('out/test_domain_support.csv',index=False)
pathlib.Path('out/uncertainty_protocol.json').write_text(json.dumps({'B':B,'type':'Rao-Wu rescaled PSU bootstrap within strata; paired common draws','scope':'Test-sample uncertainty conditional on fitted models, thresholds, and estimated labels; training/label-estimation uncertainty excluded','AUC':'point estimates only','p_values':'not computed'},indent=2));print('PAIRED INTERVALS COMPLETE',flush=True)
