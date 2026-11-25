"""Convert verified external count archives without changing source files."""

# Keep established imports and experiment-relative paths available.
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
if str(_code_root) not in _sys.path:
    _sys.path.insert(0, str(_code_root))
from benchmark_paths import CODE_ROOT, BENCH_ROOT, PROJECT_ROOT, script_path

import argparse
import io
import subprocess
import tarfile
import zipfile
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--open_path', required=True)
    p.add_argument('--save_path', required=True)
    p.add_argument('--r_path', default='Rscript')
    args = p.parse_args()
    source, dest = Path(args.open_path), Path(args.save_path)
    for name in ['GSE202234', 'GSE163836']:
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        label_table = pd.read_csv(source/name/'cell_labels.tsv', sep='\t')
        label_table.to_csv(folder/'counts_labels.tsv', sep='\t', index=False)
        output = folder / 'counts.h5ad'
        if output.exists():
            print('Existing:', output, flush=True)
            continue
        labels = label_table.set_index('Cell_ID')
        if name == 'GSE202234':
            parts = []
            with tarfile.open(source/name/'GSE202234_RAW.tar') as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    gsm = member.name.split('_')[0]
                    with zipfile.ZipFile(io.BytesIO(tar.extractfile(member).read())) as z:
                        names = z.namelist()
                        read = lambda suffix: z.read(next(n for n in names if n.endswith(suffix)))
                        x = mmread(io.BytesIO(read('matrix.mtx'))).T.tocsr()
                        ids = read('barcodes.tsv').decode().splitlines()
                        genes = pd.read_csv(io.BytesIO(read('genes.tsv')), sep='\t', header=None, dtype=str)
                        symbols = genes.iloc[:, -1].tolist()
                        unique = pd.Index(symbols).drop_duplicates()
                        mapping = sparse.csr_matrix((np.ones(len(symbols)),
                            (np.arange(len(symbols)), unique.get_indexer(symbols))),
                            shape=(len(symbols), len(unique)))
                        x = (x @ mapping).tocsr()
                        symbols = unique.tolist()
                        cells = [gsm+'_'+cell for cell in ids]
                        # Use the validated label IDs, preserving original barcode spelling.
                        sample = labels[labels.sample_id.eq(gsm)]
                        lookup = dict(zip(sample.original_cell_id, sample.index))
                        cells = [lookup[cell] for cell in ids]
                        parts.append(ad.AnnData(x, obs=labels.loc[cells].copy(),
                                                var=pd.DataFrame(index=symbols)))
            assert all(a.var_names.equals(parts[0].var_names) for a in parts)
            data = ad.concat(parts, join='inner', merge='same')
        else:
            script = script_path('export_external_counts.R')
            subprocess.run([args.r_path, str(script),
                            str(source/name/'GSE163836_IBC_sc10x_raw_counts.RData.gz'), str(folder)], check=True)
            ids = pd.read_csv(folder/'cells.tsv', header=None, sep='\t')[0].astype(str).tolist()
            genes = pd.read_csv(folder/'genes.tsv', header=None, sep='\t')[0].astype(str).tolist()
            data = ad.AnnData(mmread(folder/'counts.mtx').T.tocsr(),
                             obs=labels.loc[ids].copy(), var=pd.DataFrame(index=genes))
        assert data.obs_names.is_unique and data.var_names.is_unique
        assert set(data.obs_names) == set(labels.index)
        assert np.isfinite(data.X.data).all() and (data.X.data >= 0).all()
        assert np.equal(data.X.data, np.floor(data.X.data)).all()
        data = data[labels.index].copy()
        data.X = sparse.csr_matrix(data.X, dtype=np.float32)
        data.obs['response'] = data.obs['label']
        data.write_h5ad(output, compression='gzip')
        print(name, data.shape, data.obs.label.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()
