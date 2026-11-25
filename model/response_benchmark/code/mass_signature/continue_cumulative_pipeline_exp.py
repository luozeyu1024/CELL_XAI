"""Complete source-only GPU profiling and continue the remaining original designs."""
import os
import argparse,subprocess,sys
from pathlib import Path
from project_paths import project_root
import psutil
HERE=Path(__file__).resolve().parent;ROOT=project_root()
def run(args):
 print('RUN',args,flush=True);subprocess.run(args,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--wait_legacy_worker',type=int,default=0);a=p.parse_args()
 if a.wait_legacy_worker:
  try:
   process=psutil.Process(a.wait_legacy_worker);cmd=process.cmdline();assert '--stage' in cmd and 'worker' in cmd and any(s.endswith('train_cumulative_benchmark_exp.py') for s in cmd)
   print('WAIT_CURRENT_UNLOCKED_WORKER',a.wait_legacy_worker,flush=True);process.wait()
  except psutil.NoSuchProcess:pass
 datasets=['GSE149383_PC9_DTEP_sample_grouped','GSE165897_pre_EOC','GSE223003_carboplatin_pre','GSE223003_nk_pre','GSE223003_olaparib_pre']
 run([sys.executable,'-u',str(HERE/'benchmark_cumulative_gpu_exp.py'),'--datasets',*datasets])
 run([sys.executable,'-u',str(HERE/'select_cumulative_policies_exp.py'),'--choices',str(ROOT/'model/config/cumulative-thresholds.json')])
 run([sys.executable,'-u',str(HERE/'validate_cumulative_graphs_exp.py')])
 run([sys.executable,'-u',str(HERE/'train_cumulative_benchmark_exp.py'),'--stage','train','--datasets',*datasets])
 # Every source model must exist before frozen transfer starts.
 from cumulative_common import jobs,RELATIONS
 from train_cumulative_benchmark_exp import trial_folder
 import json,time
 while True:
  missing=[str(trial_folder(j,r)) for j in jobs() for r in RELATIONS if not (trial_folder(j,r)/'metrics.json').exists() or 'checkpoint_reload_max_margin_error' not in json.loads((trial_folder(j,r)/'metrics.json').read_text())]
  if not missing:break
  live=[]
  for proc in psutil.process_iter(['pid','name','cmdline']):
   c=proc.info['cmdline'] or []
   if proc.pid!=psutil.Process().pid and proc.info['name']=='python.exe' and any(s.endswith('train_cumulative_benchmark_exp.py') for s in c):live.append(proc)
  if not live:raise RuntimeError('Missing source trials without a live training controller: '+str(missing))
  print('WAIT_OTHER_SOURCE_CONTROLLER',len(missing),'trials',flush=True);psutil.wait_procs(live,timeout=30)
 run([sys.executable,'-u',str(HERE/'train_cumulative_benchmark_exp.py'),'--stage','transfer'])
 run([sys.executable,'-u',str(HERE/'report_cumulative_benchmark_exp.py')])
 run([sys.executable,'-u',str(HERE/'validate_cumulative_experiment_exp.py')])
 run([os.environ.get('CELL_XAI_PLOT_PYTHON', sys.executable),str(HERE.parent/'report/plot_cumulative_graphs_exp.py'),'--open_path',str(ROOT/'work/cumulative_signature'),'--save_path',str(ROOT/'work/cumulative_signature/figures')])
 run([sys.executable,str(HERE/'document_cumulative_results_exp.py')])
if __name__=='__main__':main()

