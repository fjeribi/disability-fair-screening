"""Re-run corrected experiments from verified public raw files. No archived result is consumed."""
import argparse,os,pathlib,sys,subprocess,hashlib,json,time,shutil
root=pathlib.Path(__file__).resolve().parents[1];os.chdir(root)
a=argparse.ArgumentParser();a.add_argument('--stage',choices=['all','prepare','lca','models','measurement','postprocess','paper','tests'],default='all');a.add_argument('--rscript',default=os.environ.get('RSCRIPT','Rscript'));args=a.parse_args();pathlib.Path('out').mkdir(exist_ok=True)
status=[]
def run(name,command):
 print('START',name,flush=True);start=time.time()
 with open('out/run_'+name+'.log','w',encoding='utf-8') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
 status.append(dict(step=name,returncode=p.returncode,seconds=time.time()-start));pathlib.Path('out/execution_status.json').write_text(json.dumps(status,indent=2))
 if p.returncode:raise SystemExit('Failed '+name+'; inspect out/run_'+name+'.log')
 print('COMPLETE',name,flush=True)
def py(script):run(pathlib.Path(script).stem,[sys.executable,'code/'+script])
def py_scale(script,scale):
 os.environ['SCALE']=scale;run(pathlib.Path(script).stem+'_'+scale,[sys.executable,'code/'+script]);os.environ.pop('SCALE')
if args.stage in ['all','prepare']:
 for line in pathlib.Path('data/SHA256SUMS.txt').read_text().splitlines():
  expected,name=line.split();p=pathlib.Path('data/raw')/name
  if not p.exists():raise SystemExit('Place required public raw file at '+str(p))
  if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise SystemExit('Input hash mismatch: '+name)
 for file in ['01_build_nhis.py','03_nhanes_build.py','03b_prepare_R_inputs.py']:py(file)
if args.stage in ['all','tests']:py('test_corrections.py')
if args.stage in ['all','lca']:
 py('corrected_audit.py');py('check_lca_patterns.py');py('cohort_flow.py')
if args.stage in ['all','models']:
 py('run_matched_experiments.py');py('run_corrected_primary.py');py('corrected_item_interactions.py')
 for sc in ['dep','anx']:py_scale('fullresource_baselines.py',sc)   # full-resource comparator arm (second revision)
if args.stage in ['all','measurement']:
 for file in ['04_invariance.R','13_nhanes_invariance.R']:run(pathlib.Path(file).stem,[args.rscript,'code/'+file])
if args.stage in ['all','postprocess']:
 for file in ['corrected_model_simulations.py','corrected_uncertainty.py','corrected_referral.py']:py(file)
if args.stage in ['all','paper']:
 # descriptive, robustness, tolerance, calibration and operating-point analyses; paper tables and figures
 for file in ['02_describe.py','17_robustness.py','21_dafsI_eps.py','27_sex_attrition_power.py','32_calibration.py']:py(file)
 for sc in ['dep','anx']:py_scale('33_frontier.py',sc)
 for sc in ['dep','anx']:   # narrow accuracy bands (Table 13)
  os.environ.update(SCALE=sc,BAND='0.005');run('33_frontier_band_'+sc,[sys.executable,'code/33_frontier.py']);os.environ.pop('SCALE');os.environ.pop('BAND')
 py('37_lca_sensitivity.py')   # alternative latent-class indicator specifications
 for sc in ['dep','anx']:py_scale('36_xgb_seeds.py',sc)   # seed stability of the gradient-boosted variants
 for file in ['30_paper_tables.py','34_frontier_table.py','28_flowchart.py','31_paper_figures.py']:py(file)
print('Requested experiment stages completed.',flush=True)

