"""Rank gene neighbors independently and jointly using existing embeddings."""
import argparse
import csv
import json
import platform
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


MODELS = {
    'esm2_650m': 'ESM2_650M',
    'qwen': 'Qwen',
    'state_15b': 'STATE_ESM2_15B',
}
LAYERS = ('direct', 'spectrum', 'family', 'activation', 'inactivation',
          'repair', 'resistance', 'pathway')


class GeneMapper:
    def __init__(self, path):
        self.records = {}
        self.aliases = defaultdict(set)
        self.accessions = defaultdict(set)
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if row['status'] != 'Approved':
                    continue
                gene = row['symbol']
                self.records[gene] = row
                for field in ('alias_symbol', 'prev_symbol', 'ensembl_gene_id'):
                    for item in row.get(field, '').split('|'):
                        if item:
                            self.aliases[item].add(gene)
                for accession in row.get('uniprot_ids', '').split('|'):
                    if accession:
                        self.accessions[accession].add(gene)

    def canonical(self, name):
        name = str(name).strip()
        if name in self.records:
            return name
        name = re.sub(r'^(ENSG\d+)\.\d+$', r'\1', name)
        hits = self.aliases.get(name, set())
        return next(iter(hits)) if len(hits) == 1 else None

    def protein_gene(self, accession, names):
        hits = self.accessions.get(accession, set())
        if len(hits) == 1:
            return next(iter(hits))
        named = {self.canonical(n) for n in str(names).split()}
        named.discard(None)
        hits = hits & named if hits else named
        return next(iter(hits)) if len(hits) == 1 else None


def load_model(path, mapper, model):
    explicit = {}
    mapping_path = path.parent / 'protein_gene_mapping.tsv'
    if model == 'state_15b' and mapping_path.exists():
        with mapping_path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                gene = mapper.canonical(row['hgnc_symbol'])
                accession = row['accession']
                if accession in explicit and explicit[accession] != gene:
                    raise ValueError('Conflicting STATE accession mapping: ' + accession)
                explicit[accession] = gene
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['ids'].astype(str)
        names = archive['gene_names'].astype(str)
        raw = np.asarray(archive['embeddings'], dtype=np.float32)
    if raw.ndim != 2 or len(ids) != len(raw) or len(names) != len(raw):
        raise ValueError('Invalid embedding shape: ' + str(path))
    if len(set(ids.tolist())) != len(ids):
        raise ValueError('Duplicate protein accessions: ' + str(path))
    grouped = defaultdict(list)
    skipped = []
    for i, (accession, name) in enumerate(zip(ids, names)):
        gene = explicit.get(accession) if explicit else mapper.protein_gene(accession, name)
        reason = None
        if gene is None:
            reason = 'unmapped_or_ambiguous_gene'
        elif not np.isfinite(raw[i]).all():
            reason = 'nonfinite_embedding'
        elif np.linalg.norm(raw[i]) == 0:
            reason = 'zero_embedding'
        if reason:
            skipped.append({'uniprot_id': accession, 'reason': reason})
        else:
            grouped[gene].append(i)
    genes, vectors, proteins = [], [], {}
    for gene, indices in sorted(grouped.items()):
        # Deduplicate identical raw vectors before averaging normalized proteins.
        block = raw[indices].copy()
        if len(indices) > 1:
            block = np.unique(block, axis=0)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        vector = block.mean(axis=0)
        length = np.linalg.norm(vector)
        if not np.isfinite(length) or length == 0:
            skipped.append({'gene_symbol': gene, 'reason': 'invalid_gene_centroid'})
            continue
        genes.append(gene)
        vectors.append(vector / length)
        proteins[gene] = sorted(ids[indices].tolist())
    if not genes:
        raise ValueError('No valid mapped genes: ' + str(path))
    return {'genes': genes, 'matrix': np.stack(vectors),
            'index': {g: i for i, g in enumerate(genes)}, 'proteins': proteins,
            'metadata': {'file': str(path.resolve()),
                         'n_input_proteins': len(ids), 'n_genes': len(genes),
                         'dimension': int(raw.shape[1]), 'excluded': skipped,
                         'mapping_file': str(mapping_path.resolve()) if explicit else None}}


def load_clusters(path, mapper, gene_column, cluster_column):
    if path.suffix.lower() == '.xlsx':
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        values = iter(book.active.values)
        header = list(next(values))
        rows = [dict(zip(header, row)) for row in values]
        book.close()
    else:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream, delimiter=',' if path.suffix == '.csv' else '\t')
            header = reader.fieldnames or []
            rows = list(reader)
    if gene_column not in header or cluster_column not in header:
        raise ValueError('Cluster table is missing the requested columns')
    result = {}
    for row in rows:
        gene = mapper.canonical(row[gene_column])
        label = row[cluster_column]
        if not gene or label is None or str(label).strip().lower() in ('', 'nan', 'none'):
            continue
        label = str(label)
        if gene in result and result[gene] != label:
            raise ValueError('Conflicting cluster labels for ' + gene)
        result[gene] = label
    if not result:
        raise ValueError('No usable STATE cluster assignments')
    return result


def select_seeds(data, mapper, layers, datasets):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    skipped = []
    for i, record in enumerate(data.get('targets', [])):
        dataset_value = str(record.get('dataset', ''))
        if 'sciplex' in dataset_value.lower():
            skipped.append({'record_index': i, 'reason': 'sciplex_excluded'})
            continue
        layer = record.get('layer')
        if layer not in layers:
            continue
        for dataset in (x.strip() for x in dataset_value.split(';')):
            if not dataset or (datasets and dataset not in datasets):
                continue
            # Keep selected but empty layers visible in the output.
            group = groups[dataset][layer]
            gene = mapper.canonical(record.get('gene', ''))
            reason = None
            if not record.get('gene'):
                reason = 'nonprotein_or_empty_gene'
            elif not gene:
                reason = 'unmapped_or_ambiguous_gene'
            elif dataset == 'GSE117872' and gene == 'SOX9':
                # The input explicitly restricts this record to HN120, not HN137.
                reason = 'source_record_not_applicable_to_HN137'
            if reason:
                skipped.append({'record_index': i, 'dataset_id': dataset, 'reason': reason})
            else:
                group[gene].append(i)
    return groups, skipped


def order_indices(scores, genes):
    # Stable gene-symbol tie breaking makes top-k reproducible.
    return np.lexsort((np.asarray(genes), -scores))


def cluster_annotation(seed, gene, clusters):
    a, b = clusters.get(seed), clusters.get(gene)
    return {'seed_cluster_id': a, 'candidate_cluster_id': b,
            'same_cluster': a == b if a is not None and b is not None else None}


def load_ppi(path, mapper):
    nodes, edges = set(), {}
    counts = defaultdict(int)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'g1_symbol', 'g2_symbol', 'conn'} <= set(reader.fieldnames or []):
            raise ValueError('PPI requires g1_symbol, g2_symbol and conn columns')
        for row in reader:
            counts['input_rows'] += 1
            a, b = mapper.canonical(row['g1_symbol']), mapper.canonical(row['g2_symbol'])
            if not a or not b:
                counts['unmapped_or_ambiguous_rows'] += 1
                continue
            weight = float(row['conn'])
            if not np.isfinite(weight):
                raise ValueError('Nonfinite PPI weight')
            nodes.update((a, b))
            if a == b:
                counts['self_edges_excluded'] += 1
                continue
            key = tuple(sorted((a, b)))
            edges[key] = max(edges.get(key, float('-inf')), weight)
    return {'nodes': nodes, 'edges': edges, 'metadata': {
        **counts, 'mapped_nodes': len(nodes), 'unique_undirected_edges': len(edges),
        'duplicate_weight_rule': 'maximum', 'edge_weight_filter': 'none',
        'edge_semantics': 'local_scnet_network_edge_not_verified_physical_binding'}}


def ppi_annotation(seed, gene, ppi):
    missing = [g for g in (seed, gene) if g not in ppi['nodes']]
    if missing:
        return {'is_direct_neighbor': None, 'edge_weight': None,
                'status': 'node_unavailable', 'missing_genes': missing}
    weight = ppi['edges'].get(tuple(sorted((seed, gene))))
    return {'is_direct_neighbor': weight is not None, 'edge_weight': weight,
            'status': 'both_nodes_present'}


def per_model(seed, model_name, model, clusters, threshold, top_k):
    if seed not in model['index']:
        return {'status': 'missing_seed_embedding', 'candidates': []}, None
    scores = np.clip(model['matrix'] @ model['matrix'][model['index'][seed]], -1, 1)
    ordered = [int(i) for i in order_indices(scores, model['genes'])
               if model['genes'][i] != seed]
    selected = []
    counts = defaultdict(int, eligible=0, below_or_equal_threshold=0)
    for similarity_rank, i in enumerate(ordered, 1):
        gene = model['genes'][i]
        if scores[i] <= threshold:
            counts['below_or_equal_threshold'] += 1
            continue
        counts['eligible'] += 1
        selected.append({'gene_symbol': gene, 'uniprot_ids': model['proteins'][gene],
                             'rank': len(selected) + 1, 'similarity_rank': similarity_rank,
                             'cosine_similarity': float(scores[i]),
                             'state_cluster': cluster_annotation(seed, gene, clusters)})
    return {'status': 'completed' if selected else 'no_candidates_above_threshold',
            'seed_uniprot_ids': model['proteins'][seed],
            'n_scored': len(ordered), 'counts': dict(counts),
            'candidates': selected[:top_k], 'all_threshold_candidates': selected}, scores


def fuse(seed, models, scores_by_model, common_genes, clusters, threshold, top_k, rrf_k):
    missing = [m for m, scores in scores_by_model.items() if scores is None]
    if missing:
        return {'status': 'missing_seed_embedding', 'missing_models': missing, 'candidates': []}
    genes = [g for g in common_genes if g != seed]
    if not genes:
        return {'status': 'empty_common_universe', 'candidates': []}
    ranks, similarities = {}, {}
    for name, model in models.items():
        scores = np.asarray([scores_by_model[name][model['index'][g]] for g in genes])
        ordered = order_indices(scores, genes)
        rank = np.empty(len(genes), dtype=np.int64)
        rank[ordered] = np.arange(1, len(genes) + 1)
        ranks[name], similarities[name] = rank, scores
    fused = np.mean([1.0 / (rrf_k + rank) for rank in ranks.values()], axis=0)
    selected = []
    counts = defaultdict(int, eligible=0, below_or_equal_threshold=0)
    for i in order_indices(fused, genes):
        gene = genes[i]
        if not all(values[i] > threshold for values in similarities.values()):
            counts['below_or_equal_threshold'] += 1
            continue
        counts['eligible'] += 1
        selected.append({'gene_symbol': gene, 'rank': len(selected) + 1,
                             'rrf_score': float(fused[i]),
                             'cosine_by_model': {m: float(s[i]) for m, s in similarities.items()},
                             'common_universe_rank_by_model': {m: int(r[i]) for m, r in ranks.items()},
                             'uniprot_ids_by_model': {m: v['proteins'][gene] for m, v in models.items()},
                             'state_cluster': cluster_annotation(seed, gene, clusters)})
    return {'status': 'completed' if selected else 'no_candidates_above_threshold', 'models': list(models),
            'n_scored': len(genes), 'counts': dict(counts),
            'candidates': selected[:top_k], 'all_threshold_candidates': selected}


def support_summary(seed_results, names, full=False):
    summaries = {}
    for name in list(names) + ['fusion']:
        supported = defaultdict(list)
        for result in seed_results:
            ranking = result['fusion'] if name == 'fusion' else result['per_model'][name]
            for row in ranking.get('all_threshold_candidates' if full else 'candidates', []):
                supported[row['gene_symbol']].append(result['seed_gene'])
        summaries[name] = [{'gene_symbol': g, 'supporting_seed_count': len(set(seeds)),
                            'supporting_seeds': sorted(set(seeds))}
                           for g, seeds in sorted(supported.items(), key=lambda x: (-len(set(x[1])), x[0]))]
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--open_path', type=Path, required=True)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--hgnc_path', type=Path, required=True)
    parser.add_argument('--save_path', type=Path, required=True, help='One output JSON file')
    parser.add_argument('--layer_vocabulary', nargs='+', choices=['all'] + list(LAYERS), default=['direct'])
    parser.add_argument('--model', nargs='+', choices=['all'] + list(MODELS), default=['all'])
    parser.add_argument('--dataset', nargs='+', help='Exact dataset IDs; omit to select all eligible datasets')
    parser.add_argument('--similarity_threshold', type=float, default=0.8)
    parser.add_argument('--ppi_path', type=Path)
    parser.add_argument('--state_cluster_path', type=Path)
    parser.add_argument('--state_gene_column', default='gene_id')
    parser.add_argument('--state_cluster_column', default='cluster')
    parser.add_argument('--top_k', type=int, default=100, help='Top ranking view; full threshold candidates are always retained')
    parser.add_argument('--rrf_k', type=float, default=60.0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    for values in (args.layer_vocabulary, args.model):
        if 'all' in values and len(values) != 1:
            parser.error('Use all alone, not together with individual names')
    if args.top_k < 1 or not np.isfinite(args.rrf_k) or args.rrf_k <= 0:
        parser.error('top_k and rrf_k must be positive and finite')
    args.models = list(MODELS) if args.model == ['all'] else list(dict.fromkeys(args.model))
    args.layers = list(LAYERS) if args.layer_vocabulary == ['all'] else list(dict.fromkeys(args.layer_vocabulary))
    if not np.isfinite(args.similarity_threshold) or not -1 <= args.similarity_threshold <= 1:
        parser.error('similarity_threshold must be finite and within [-1, 1]')
    args.state_cluster_path = args.state_cluster_path or args.data_path / 'state_gene_knn_leiden_clusters.xlsx'
    args.ppi_path = args.ppi_path or args.data_path / 'format_h_sapiens.csv'
    if args.save_path.suffix.lower() != '.json':
        parser.error('save_path must name a JSON file')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.save_path.exists() and not args.overwrite:
        raise FileExistsError('Output already exists; use --overwrite to replace it')
    if args.save_path.resolve() == args.open_path.resolve():
        raise ValueError('Output must not overwrite the input JSON')
    data = json.loads(args.open_path.read_text(encoding='utf-8-sig'))
    if not isinstance(data.get('targets'), list):
        raise ValueError('Input must contain a targets list')
    mapper = GeneMapper(args.hgnc_path)
    groups, skipped = select_seeds(data, mapper, args.layers, args.dataset)
    if not groups:
        raise ValueError('No dataset/layer groups matched the selection')
    clusters = load_clusters(args.state_cluster_path, mapper, args.state_gene_column, args.state_cluster_column)
    ppi = load_ppi(args.ppi_path, mapper)
    models = {}
    for name in args.models:
        print('Loading', name, flush=True)
        models[name] = load_model(args.data_path / MODELS[name] / 'human_swissprot_aligned.npz', mapper, name)
    common = sorted(set.intersection(*(set(v['genes']) for v in models.values())))
    results = []
    cached = {}
    for dataset, layers in sorted(groups.items()):
        layer_results = []
        for layer, seeds in sorted(layers.items()):
            seed_results = []
            for seed, records in sorted(seeds.items()):
                if seed not in cached:
                    individual, scores = {}, {}
                    for name, model in models.items():
                        individual[name], scores[name] = per_model(seed, name, model, clusters,
                            args.similarity_threshold, args.top_k)
                    fusion = fuse(seed, models, scores, common, clusters,
                                  args.similarity_threshold, args.top_k, args.rrf_k)
                    for result in list(individual.values()) + [fusion]:
                        for row in result.get('all_threshold_candidates', []):
                            row['ppi'] = ppi_annotation(seed, row['gene_symbol'], ppi)
                    cached[seed] = {'per_model': individual, 'fusion': fusion}
                seed_results.append({'seed_gene': seed, 'source_record_indices': records, **cached[seed]})
            layer_results.append({'layer': layer, 'status': 'completed' if seeds else 'no_valid_seed_genes',
                                  'seed_genes': sorted(seeds), 'seed_results': seed_results,
                                  'gene_set_summary': support_summary(seed_results, args.models),
                                  'gene_set_summary_full': support_summary(seed_results, args.models, full=True)})
        results.append({'dataset_id': dataset, 'layers': layer_results})
    output = {
        'schema_version': '2.1',
        'parameters': {'layers': args.layers, 'models': args.models, 'top_k': args.top_k,
                       'rrf_k': args.rrf_k, 'similarity_threshold': args.similarity_threshold,
                       'threshold_operator': '>', 'cluster_usage': 'annotation_only',
                       'full_threshold_pool': 'all_threshold_candidates',
                       'ppi_usage': 'annotation_only',
                       'independent_universe': 'each_model_available_genes',
                       'fusion_universe': 'intersection_of_selected_models',
                       'fusion_ranks': 'before_threshold_within_common_universe',
                       'aggregation': 'normalize_mean_of_unique_normalized_protein_vectors',
                       'self_match': 'excluded', 'sciplex': 'excluded'},
        'provenance': {'input_file': str(args.open_path.resolve()),
                       'hgnc_file': str(args.hgnc_path.resolve()),
                       'python': platform.python_version(),
                       'numpy': np.__version__,
                       'ppi_file': str(args.ppi_path.resolve()),
                       'ppi_mapping': ppi['metadata'],
                       'state_cluster_file': str(args.state_cluster_path.resolve()) if clusters else None,
                       'state_gene_column': args.state_gene_column if clusters else None,
                       'state_cluster_column': args.state_cluster_column if clusters else None},
        'model_inputs': {name: model['metadata'] for name, model in models.items()},
        'gene_vocabulary': {name: model['genes'] for name, model in models.items()},
        'n_common_candidate_genes': len(common), 'datasets': results,
        'skipped_records': skipped,
        'excluded_sections': ['sciplex3', 'perturbation'],
        'interpretation': 'Representation similarity, not gene attribution or causal target validation.'}
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize fully before touching the destination; reject nonstandard NaN values.
    text = json.dumps(output, ensure_ascii=True, indent=2, allow_nan=False)
    args.save_path.write_text(text + '\n', encoding='utf-8')
    print('Saved', args.save_path, 'datasets', len(results), 'unique seeds', len(cached))


if __name__ == '__main__':
    main()
