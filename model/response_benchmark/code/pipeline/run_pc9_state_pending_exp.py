"""Finish the PC9 sample-held-out experiment after verified SE inference."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse,json,subprocess,sys
from pathlib import Path
import psutil
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--open_path',required=True);p.add_argument('--wait_pid',type=int,default=0);a=p.parse_args()
code=CODE_ROOT;task='GSE149383_PC9_DTEP_sample_grouped'
if a.wait_pid:
 try:
  process=psutil.Process(a.wait_pid);cmd=process.cmdline()
  if not any(x.endswith('pc9_state_grouped_exp.py') for x in cmd) or '--stage' not in cmd or cmd[cmd.index('--stage')+1]!='state':raise ValueError('Unexpected SE wait process')
  print('WAIT_PC9_SE',process.pid,flush=True);process.wait()
 except psutil.NoSuchProcess:pass
validation=code.parent/'data/state_embeddings'/task/'validation.json'
if json.loads(validation.read_text())['status']!='passed':raise ValueError('SE validation incomplete')
for stage in ['cohorts','train']:
 subprocess.run([sys.executable,'-X','utf8',str(script_path('pc9_state_grouped_exp.py')),'--open_path',a.open_path,'--stage',stage],check=True)
subprocess.run([sys.executable,'-X','utf8',str(script_path('report_pc9_state_exp.py'))],check=True)
