"""Empirical referral-set coverage: OOF replication and separate held-out calibration.
Survey-weighted/class-conditional quantiles are empirical; no distribution-free guarantee is claimed.
"""
import os,sys,pathlib,pickle,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
r=pathlib.Path(__file__).resolve().parents[1];os.chdir(r);sys.path.insert(0,'code');from fairmetrics import DOMS
from statutils import bootstrap_weights
rows=[];B=500;alpha=.10
for scale in ['dep','anx']:
 d=pd.read_parquet('data/proc/ml_'+scale+'.parquet');trall=d[d.split=='train'].copy();te=d[d.split=='test'].reset_index(drop=True);suf='' if scale=='dep' else '_anx';items=[('phq8' if scale=='dep' else 'gad7')+str(i) for i in range(1,9 if scale=='dep' else 8)];biased=['phq84','phq87'] if scale=='dep' else ['gad71'];rng=np.random.default_rng(11);ps=trall.psu.unique();val=set(rng.choice(ps,int(.25*len(ps)),replace=False));tr=trall[~trall.psu.isin(val)];va=trall[trall.psu.isin(val)]
 def X(t):
  z=t[items].copy()
  for g in DOMS:
   z['d_'+g]=t['d_'+g]
   for it in biased:z[it+'x'+g]=t[it]*t['d_'+g]
  return z.to_numpy(float)
 m=LogisticRegression(C=1,max_iter=5000).fit(X(tr),tr.yA,sample_weight=tr.wt_pooled/tr.wt_pooled.mean());pv=m.predict_proba(X(va))[:,1];pt=m.predict_proba(X(te))[:,1];sc=pickle.load(open(f'out/dafsI_scores{suf}.pkl','rb'))
 for arm,cal,pcal,pte in [('OOF',trall,sc['oof'],sc['pte']),('holdout',va,pv,pt)]:
  def qhat(mask,c):
   mm=mask&(cal.yA.to_numpy()==c);v=(1-pcal if c==1 else pcal)[mm];w=cal.wt_pooled.to_numpy()[mm];o=np.argsort(v);level=min(1,np.ceil((len(v)+1)*(1-alpha))/len(v));return v[o][min(np.searchsorted(np.cumsum(w[o])/w.sum(),level),len(v)-1)]
  for grouping in ['marginal','priority-group']:
   q={g:[qhat(np.ones(len(cal),bool) if grouping=='marginal' else cal.fgroup.to_numpy()==g,c) for c in [0,1]] for g in cal.fgroup.unique()};qq=np.array([q[g] for g in te.fgroup]);in0=pte<=qq[:,0];in1=1-pte<=qq[:,1];y=te.yA.to_numpy();cover=np.where(y==1,in1,in0);refer=in0==in1;dec=np.where(in1,1,0);rng=np.random.default_rng(82);bs=np.array([bootstrap_weights(te,rng) for _ in range(B)])
   for group,mask in [('none',te.disab.to_numpy()==0)]+[(g,te['d_'+g].to_numpy()==1) for g in DOMS]:
    for metric,v,mm in [('coverage',cover,mask),('coverage_cases',cover,mask&(y==1)),('referral_rate',refer,mask),('accuracy_decided',dec==y,mask&~refer)]:
     w=te.wt_pooled.to_numpy()*mm;rep=(bs@(v*mm))/(bs@mm);lo,hi=np.nanquantile(rep,[.025,.975]);rows.append(dict(scale=scale,calibration=arm,grouping=grouping,group=group,metric=metric,estimate=np.sum(w*v)/w.sum(),lo=lo,hi=hi,n=int(mm.sum()),B=B,alpha=alpha))
  print(scale,arm,'referral complete',flush=True)
pd.DataFrame(rows).to_csv('out/corrected_referral_coverage.csv',index=False);print('REFERRAL ANALYSIS COMPLETE',flush=True)
