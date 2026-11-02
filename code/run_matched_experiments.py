import os,subprocess,sys,pathlib,time,json
r=pathlib.Path(__file__).resolve().parents[1];os.chdir(r);steps=[]
for scale in ['dep','anx']:
 for target in ['yA','yB']:
  for script in ['08_baselines.py','matched_proposed.py']:
   label=f'{scale}_{target}_{script[:-3]}';env=os.environ.copy();env['SCALE']=scale;t=time.time();print('START',label,flush=True)
   with open('out/'+label+'.log','w',encoding='utf-8') as f:p=subprocess.run([sys.executable,'code/'+script,target],env=env,stdout=f,stderr=subprocess.STDOUT)
   steps.append(dict(step=label,returncode=p.returncode,seconds=time.time()-t));pathlib.Path('out/matched_run_status.json').write_text(json.dumps(steps,indent=2));print('FINISH',label,p.returncode,flush=True)
   if p.returncode:raise RuntimeError(label)
