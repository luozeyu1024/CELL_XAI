"""Locate the analysis project when scripts run from a project or installed skill."""
import os
from pathlib import Path

def project_root():
 configured=os.environ.get('CELL_XAI_PROJECT_ROOT')
 if configured:
  root=Path(configured).expanduser().resolve()
  if not (root/'model/response_benchmark/code/core/graph_model.py').is_file():raise FileNotFoundError('CELL_XAI_PROJECT_ROOT does not contain the model code: '+str(root))
  import sys
  if str(root/'model/response_benchmark/code') not in sys.path:sys.path.insert(0,str(root/'model/response_benchmark/code'))
  import benchmark_paths
  return root
 for origin in [Path(__file__).resolve().parent,Path.cwd()]:
  for root in [origin,*origin.parents]:
   if (root/'model/response_benchmark/code/core/graph_model.py').is_file():
    import sys
    if str(root/'model/response_benchmark/code') not in sys.path:sys.path.insert(0,str(root/'model/response_benchmark/code'))
    import benchmark_paths
    return root
 raise FileNotFoundError('Run inside the project or set CELL_XAI_PROJECT_ROOT to its root directory.')
