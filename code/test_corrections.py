import sys,ast,pathlib,numpy as np,pandas as pd
root=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'code'));from statutils import roc_pts,bootstrap_weights,lca2
from fairmetrics import group_rates
s=np.array([2,2,1,1]);y=np.array([1,0,1,0]);w=np.array([1.,2,3,4]);a=roc_pts(s,y,w)
for perm in [np.array([1,0,3,2]),np.array([3,2,1,0])]:
 b=roc_pts(s[perm],y[perm],w[perm]);assert all(np.allclose(x,z) for x,z in zip(a,b))
for t,tp,fp in zip(*a):
 actual=group_rates(y,(s>=t).astype(float),w);assert np.allclose(actual[:2],[tp,fp])
assert np.allclose(group_rates(y,np.full(4,.25),w),[.25,.25,.25])
d=pd.DataFrame({'stratum':[1,1,2,2],'psu':['a','b','c','d'],'wt_pooled':[1.,1,1,1]})
assert bootstrap_weights(d,np.random.default_rng(9)).sum()==4
old=pathlib.Path(root/'code/reference_eo_function.py').read_text();tree=ast.parse((root/'code/09_proposed.py').read_text());new=ast.get_source_segment((root/'code/09_proposed.py').read_text(),next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='eo_thresholds'))
rng=np.random.default_rng(62);g=np.repeat(['none','cognition','vision'],40);s=np.round(rng.random(120),1);y=np.tile([0,1],60);w=rng.uniform(.5,2,120)
ns={'np':np,'roc_pts':roc_pts,'GROUPS':np.unique(g)};exec(old,ns);a=ns['eo_thresholds'](s,y,w,g,.2);exec(new,ns);b=ns['eo_thresholds'](s,y,w,g,.2);assert a==b,(a,b)
print('PASS: tied ROC attainability and permutation invariance; randomized decision metrics; PSU rescaling; vectorized threshold equivalence')

