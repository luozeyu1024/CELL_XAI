"""Run the complete independent-replicate tasks sequentially on the existing GPU."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
from pathlib import Path
import subprocess
import sys
import psutil
from mechanism_grouped_exp import primary_counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True)
    parser.add_argument('--tasks',nargs='+',required=True)
    parser.add_argument('--wait_pid',type=int,default=0)
    parser.add_argument('--stage',choices=['prepare','all'],default='prepare')
    args=parser.parse_args();root=Path(args.open_path)
    for task in args.tasks:
        print('PREPARE_PRIMARY_COUNTS',task,flush=True)
        primary_counts(root,task)
    if args.wait_pid:
        try:
            process=psutil.Process(args.wait_pid)
            if not any(str(x).endswith('mechanism_grouped_exp.py') for x in process.cmdline()):
                raise ValueError('Wait PID does not identify the preceding grouped pipeline')
            print('WAIT_PREVIOUS_GPU_PIPELINE',args.wait_pid,flush=True);process.wait()
        except psutil.NoSuchProcess: pass
    for task in args.tasks:
        subprocess.run([sys.executable,str(script_path('mechanism_grouped_exp.py')),
                        '--open_path',str(root),'--task',task,'--stage',args.stage],check=True)


if __name__=='__main__': main()
