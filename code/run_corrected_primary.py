"""Run corrected full-training cross-fitted primary analyses; no archived predictions are read."""
import os,sys,subprocess,pathlib,pickle,json,time
r=pathlib.Path(__file__).resolve().parents[1];os.chdir(r)
status=[]
for scale in ['dep','anx']:
 os.environ['SCALE']=scale
 scripts=['09_proposed.py'] if scale=='dep' else ['25_anx_item_dp.py','09_proposed.py']
 for script in scripts:
  print('START',scale,script,flush=True); t=time.time()
  with open(f'out/primary_{scale}_{script}.log','w') as f:p=subprocess.run([sys.executable,'code/'+script,'yA','0.05'],stdout=f,stderr=subprocess.STDOUT)
  status.append(dict(scale=scale,script=script,returncode=p.returncode,seconds=time.time()-t))
  pathlib.Path('out/primary_status.json').write_text(json.dumps(status,indent=2))
  if p.returncode:raise RuntimeError((scale,script))
 print('START',scale,'DAFS-I',flush=True)
 src=pathlib.Path('code/18_dafs_variants.py').read_text().split('pp = pickle.load')[0]
 ns={'__name__':'__main__'};exec(compile(src,'18_dafs_variants.py','exec'),ns)
 suf=ns['SUF'];pickle.dump({'DAFS-I':(ns['pte'],ns['yh'])},open(f'out/pred_dafsI{suf}.pkl','wb'))
 pickle.dump({k:ns[k] for k in ['oof','pte','thr']},open(f'out/dafsI_scores{suf}.pkl','wb'))
 ns['pd'].DataFrame([dict(method='DAFS-I',**ns['r'])]).to_csv(f'out/primary_dafsI{suf}.csv',index=False)
 pathlib.Path(f'out/primary_thresholds{suf}.json').write_text(json.dumps({k:float(v) for k,v in ns['thr'].items()},indent=2))
 print('FINISH',scale,'DAFS-I',flush=True)
print('PRIMARY COMPLETE',flush=True)
