"""Descriptive CFA simulations, including identical reference measurement parameters.
No subtraction is interpreted causally. NHANES merged scores are not PHQ-8 cutoffs.
"""
import os,pathlib,json,copy,numpy as np,pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
r=pathlib.Path(__file__).resolve().parents[1];os.chdir(r)
N=400000;rows=[];skips=[]
def a(g,k):return np.asarray(g[k],float).reshape(-1)
def simulate(g,seed):
 rng=np.random.default_rng(seed);lam=a(g,'lambda');tau=a(g,'tau').reshape(len(lam),-1);theta=a(g,'theta');nu=a(g,'nu');eta=rng.normal(a(g,'alpha')[0],np.sqrt(a(g,'psi')[0]),N)
 z=nu+eta[:,None]*lam+rng.normal(size=(N,len(lam)))*np.sqrt(theta);return eta,(z[:,:,None]>tau[None]).sum(axis=2).sum(axis=1)
def expected_cut(g,score):
 lam=a(g,'lambda');tau=a(g,'tau').reshape(len(lam),-1);theta=a(g,'theta');nu=a(g,'nu')
 def f(x):return norm.cdf((nu[:,None]+lam[:,None]*x-tau)/np.sqrt(theta[:,None])).sum()-score
 return brentq(f,-20,20)
for survey,path in [('NHIS','invariance_params.json'),('NHANES','nhanes_invariance_params.json')]:
 P=json.load(open('out/'+path));d=pd.read_csv('data/proc/'+('nhis' if survey=='NHIS' else 'nhanes')+'_analytic.csv');d=d[d.split=='train'];d['d_any']=d.disab
 keys=sorted({k.rsplit('|',1)[0] for k in P})
 for key in keys:
  scale,domain=key.split('|');scalar=P.get(key+'|scalar')
  if scalar is None:skips.append(dict(survey=survey,scale=scale,domain=domain,reason='No converged admissible scalar model'));continue
  obs=d[(d.disab==0)|(d['d_'+domain]==1)].dropna(subset=[('phq8' if scale=='phq' else 'gad7')+str(i) for i in range(1,9 if scale=='phq' else 8)])
  share=obs.loc[obs['d_'+domain]==1,'wt_pooled'].sum()/obs.wt_pooled.sum()
  if survey=='NHIS':cut=10;latent=expected_cut(scalar['nodisab'],cut)
  else:
   temp={g:simulate(scalar[g],610+i) for i,g in enumerate(['domain','nodisab'])};prop=np.average(obs.phq_pos,weights=obs.wt_pooled)
   cuts=[c for c in range(18) if share*np.mean(temp['domain'][1]>=c)+(1-share)*np.mean(temp['nodisab'][1]>=c)<=prop]
   if not cuts:skips.append(dict(survey=survey,scale=scale,domain=domain,reason='No merged score cut'));continue
   cut=cuts[0]
   if cut>=16:skips.append(dict(survey=survey,scale=scale,domain=domain,reason='Degenerate merged score cutoff'));continue
   latent=expected_cut(scalar['nodisab'],cut)
  models={m:P[key+'|'+m] for m in ['configural','scalar','strict'] if key+'|'+m in P}
  common=copy.deepcopy(scalar)
  for field in ['lambda','tau','theta','nu']:common['domain'][field]=copy.deepcopy(scalar['nodisab'][field])
  models['scalar_common_reference']=common
  for model,pars in models.items():
   # Common random numbers reduce Monte Carlo noise for scalar vs common measurement.
   cuteta=latent if model in ['scalar','scalar_common_reference'] else expected_cut(pars['nodisab'],cut)
   for i,g in enumerate(['domain','nodisab']):
    eta,score=simulate(pars[g],2026+i);truth=eta>cuteta;positive=score>=cut
    if not truth.any() or truth.all():skips.append(dict(survey=survey,scale=scale,domain=domain,model=model,group=g,reason='Degenerate simulated latent class'));continue
    fpr=np.mean(positive[~truth]);sens=np.mean(positive[truth]);rows.append(dict(survey=survey,scale=scale,domain=domain,model=model,group=g,score_cut=cut,latent_cut=cuteta,latent_prevalence=truth.mean(),FPR=fpr,sensitivity=sens,FPR_MCSE=np.sqrt(fpr*(1-fpr)/(~truth).sum()),N=N,clinical_cutoff_interpretation=survey=='NHIS'))
  print(survey,scale,domain,'complete',flush=True)
pd.DataFrame(rows).to_csv('out/corrected_model_simulations.csv',index=False);pathlib.Path('out/model_simulation_exclusions.json').write_text(json.dumps(skips,indent=2));print('SIMULATIONS COMPLETE',flush=True)
