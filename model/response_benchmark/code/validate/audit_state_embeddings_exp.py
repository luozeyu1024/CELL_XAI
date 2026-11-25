"""Inspect D/E array inventory for native STATE SE cell embeddings, without loading X."""

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
import h5py
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path', required=True)
    p.add_argument('--save_path', required=True)
    a = p.parse_args()
    paths = Path(a.open_path).read_text(encoding='utf-8-sig').splitlines()
    rows = []
    for name in paths:
        f = Path(name)
        if f.suffix.lower() == '.h5ad':
            try:
                with h5py.File(f, 'r') as h:
                    for key, item in h.get('obsm', {}).items():
                        if 'state' in key.lower():
                            shape = tuple(item.shape) if isinstance(item, h5py.Dataset) else tuple(item.attrs.get('shape', []))
                            rows.append(dict(path=name, format='h5ad', key=key, shape=str(shape),
                                classification='native_SE_candidate', status='readable'))
            except Exception as e:
                rows.append(dict(path=name, format='h5ad', key='', shape='', classification='unreadable', status=str(e)))
        elif f.suffix.lower() == '.npy' and 'state' in name.lower():
            try:
                x = np.load(f, mmap_mode='r', allow_pickle=False)
                if 'Protein_embedding_to_cell_embedding' in name or 'top_gene_cell_embedding' in f.name:
                    kind = 'expression_weighted_or_cluster_representation'
                elif 'dataset_embeddings' in f.name:
                    kind = 'dataset_correction_not_cell_embedding'
                elif x.ndim==2 and x.shape[1] in [2048,2058] and ('SE_inference' in name or 'env\\state\\test' in name):
                    kind = 'native_SE_candidate'
                else:
                    kind = 'unverified_or_noncell_array'
                rows.append(dict(path=name, format='npy', key='', shape=str(x.shape), classification=kind, status='readable'))
            except Exception as e:
                rows.append(dict(path=name, format='npy', key='', shape='', classification='unreadable', status=str(e)))
    out = Path(a.save_path)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'state_array_audit.tsv', sep='\t', index=False)
    summary = {'array_paths_scanned': len(paths), 'h5ad_metadata_scanned': sum(s.lower().endswith('.h5ad') for s in paths),
        'native_SE_h5ad': [r for r in rows if r['format']=='h5ad' and r['classification']=='native_SE_candidate'],
        'scope': 'Accessible files on D and E; drive permission failures are recorded separately; shape alone does not establish provenance or cell order.'}
    (out/'state_scan_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
