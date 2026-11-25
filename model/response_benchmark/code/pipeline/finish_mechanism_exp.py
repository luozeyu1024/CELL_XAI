"""Validate completed grouped and frozen tasks, then regenerate their shared statistics and PDFs."""
import os
import sys

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import json
from pathlib import Path
import subprocess
import psutil

MODELS=['state_15b','esm2_650m','qwen','ppi','no_edges','random_signature','native_state_embedding']
EXPECTED={'GSE165897_pre_EOC':4,'GSE223003_carboplatin_pre':2,'GSE223003_olaparib_pre':2,'GSE223003_nk_pre':2}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path',required=True)
    parser.add_argument('--save_path',required=True,help='Existing shared figure directory')
    parser.add_argument('--wait_pids',nargs='*',type=int,default=[])
    args=parser.parse_args();root=Path(args.open_path);figures=Path(args.save_path);code=root/'code'
    known={'mechanism_grouped_exp.py','mechanism_batch_exp.py','alk_frozen_transfer_exp.py','ovarian_frozen_transfer_exp.py'}
    for pid in args.wait_pids:
        try:
            process=psutil.Process(pid)
            if not any(Path(value).name in known for value in process.cmdline()): raise ValueError('Unexpected wait process')
            print('WAIT_EXPERIMENT',pid,flush=True);process.wait()
        except psutil.NoSuchProcess: pass
    for task,folds in EXPECTED.items():
        for fold in range(1,folds+1):
            for model in MODELS:
                kind='state_mlp' if model=='native_state_embedding' else 'gat_gine'
                folder=root/'output/grouped_adaptive'/task/f'fold_{fold}'/model/'within'/task/kind
                for name in ['checkpoint.pt','metrics.json','predictions.tsv','run_parameters.json']:
                    if not (folder/name).exists(): raise ValueError('Incomplete required experiment: '+str(folder/name))
                parameters=json.loads((folder/'run_parameters.json').read_text())
                policy=root.parents[1]/'gene_sim_pathway/output/signature_gene_selection_grouped'/task/f'fold_{fold}'/'dataset_policy.json'
                selected=json.loads(policy.read_text())['datasets'][task]['k']
                if parameters['selected_k']!=selected: raise ValueError('Saved model K differs from reviewed policy')
    import pandas as pd
    for name in ['alk_frozen_transfer_metrics.tsv','ovarian_frozen_transfer_metrics.tsv']:
        table=pd.read_csv(root/'output/summary'/name,sep='\t')
        if set(table.model)!=set(MODELS) or len(table)!=14 or table.duplicated(['model','scope']).any():
            raise ValueError('Incomplete or duplicated frozen prediction table: '+name)
    python=sys.executable
    subprocess.run([python,str(script_path('audit_grouped_inputs_exp.py')),'--tasks',*EXPECTED,
        '--save_path',str(root/'output/summary/grouped_input_audit.tsv')],check=True)
    subprocess.run([python,str(script_path('audit_frozen_predictions_exp.py')),'--open_path',str(root),
        '--save_path',str(root/'output/summary/frozen_prediction_audit.tsv')],check=True)
    subprocess.run([python,str(script_path('report_mechanism_exp.py')),'--open_path',str(root)],check=True)
    subprocess.run([os.environ.get('CELL_XAI_PLOT_PYTHON', sys.executable),str(script_path('plot_mechanism_exp.py')),
        '--open_path',str(root/'output/summary/mechanism_combined_metrics.tsv'),'--save_path',str(figures)],check=True)
    print('COMPLETED_REQUIRED_GROUPED_MODELS',sum(EXPECTED.values())*len(MODELS),flush=True)
    print('COMPLETED_FROZEN_MODEL_SCOPE_ROWS',28,flush=True)


if __name__=='__main__': main()
