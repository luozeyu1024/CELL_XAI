"""Map existing protein vectors to the exact global graph gene vocabulary."""

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
import sys
import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT/'gene_sim_pathway/code'))
from gene_target_neighbors_exp import GeneMapper, load_model, MODELS


def prepare(save_path, graph_path, protein_path, hgnc_path):
    save_path.mkdir(parents=True, exist_ok=True)
    vocab = pd.read_csv(graph_path/'gene_vocab.tsv', sep='\t')
    genes = vocab.gene_symbol.tolist()
    mapper = GeneMapper(hgnc_path)
    rows = []
    for name, folder in MODELS.items():
        result = load_model(protein_path/folder/'human_swissprot_aligned.npz', mapper, name)
        dimension = result['matrix'].shape[1]
        matrix = np.zeros((len(genes), dimension), dtype=np.float32)
        mapped = np.array([gene in result['index'] for gene in genes])
        for index, gene in enumerate(genes):
            if mapped[index]:
                # Unit RMS removes the 1/sqrt(d) attenuation at the input projection.
                matrix[index] = result['matrix'][result['index'][gene]] * np.sqrt(dimension)
            rows.append(dict(model=name, gene_id=index, gene_symbol=gene, mapped=bool(mapped[index]),
                             uniprot_ids='|'.join(result['proteins'].get(gene, []))))
        np.save(save_path/f'{name}.npy', matrix)
        pd.DataFrame({'gene_id':np.arange(len(genes)), 'gene_symbol':genes, 'mapped':mapped}).to_csv(
            save_path/f'{name}_coverage.tsv', sep='\t', index=False)
        record = dict(model=name, dimension=dimension, n_graph_genes=len(genes), n_mapped=int(mapped.sum()),
                      source=result['metadata']['file'], mapping_file=result['metadata']['mapping_file'],
                      protein_aggregation='deduplicate identical vectors; L2 normalize proteins; mean within gene; L2 normalize gene',
                      input_scaling='multiply L2-unit gene vector by sqrt(dimension), giving unit RMS per gene',
                      missing='zero placeholder; any missing gene actually used by a cell raises an error',
                      embedding_trainable=False)
        (save_path/f'{name}.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print(name, dimension, int(mapped.sum()), '/', len(genes), flush=True)
    vocab.to_csv(save_path/'gene_vocab.tsv', sep='\t', index=False)
    pd.DataFrame(rows).to_csv(save_path/'protein_gene_mapping.tsv', sep='\t', index=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph_path', type=Path, default=ROOT/'work/cell_graphs')
    parser.add_argument('--protein_path', type=Path, default=ROOT/'assets/proteins')
    parser.add_argument('--hgnc_path', type=Path, default=ROOT/'assets/hgnc_complete_set.txt')
    parser.add_argument('--save_path', type=Path, default=BENCH_ROOT/'data/gene_embeddings')
    args=parser.parse_args()
    prepare(args.save_path, args.graph_path, args.protein_path, args.hgnc_path)
