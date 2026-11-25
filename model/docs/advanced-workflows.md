# Grouped experiments, ablations and explanations

Use the split registry to recover source folds and external scopes. Each grouped fold carries its own graph package, selected K, source scaling statistics and checkpoint. Preserve the fold directory when reusing prepared inputs.

| Workflow | Runner command | Required inputs |
| --- | --- | --- |
| Grouped response | `grouped` | Cohort counts, SCimilarity attribution, graph resources and biological group metadata |
| PC9 state | `pc9` | PC9 counts, DTEP state labels and predefined sample folds |
| Edge ablation | `no-edges` | Prepared source and target graphs, with a source GATv2/GINE checkpoint defining training settings |
| Signature ablation | `random-signature` | Reference graphs and matched random-node packages |
| Cumulative signatures | `cumulative` | Selected per-fold policies and cumulative graph packages |
| Cumulative edge ablation | `cumulative-no-edges` | Cumulative packages and matched complete-graph checkpoints |
| Frozen graph explanations | `explain` | Trained checkpoints, prepared graph cohorts and biological group metadata |

The edge ablation preserves cell IDs, expression values and gene identity features while removing all edges. Train and transfer the ablation with the same source split and comparator settings.

For the edge-ablation command, prepare the source GSE223779 and targets GSE140440, GSE163836, GSE202234 and GSE223779_TS485T in the same run directory. The included [fixed-K policy](../../gene_sim_pathway/config/fixed-k.json) covers these cohorts. To use the validation and reporting stages, also train the complete-graph references for all four relations.

```sh
python model/run.py no-edges --open_path work/response --stage run
python model/run.py explain --open_path work/response --datasets GSE223779_TS485T --models state_15b --cells_per_group 32 --steps 64 --max_steps 512 --gnn_epochs 100
```

The external explanation command uses the source checkpoint and target cells grouped by drug and response class. Signed IG explains the class-one minus class-zero logit using zero expression and source-training sensitive-expression baselines. GNNExplainer assigns masks to existing edges. The two outputs answer different model questions and retain their respective numerical diagnostics.

### Random signature control

Use the same source and targets as the edge ablation. The sampler uses eligible expressed genes and matches the reference node count, edge density, degree distribution and isolated fraction within the configured tolerances.

```sh
python gene_sim_pathway/code/state_signature_ablation/build_random_state_signatures_exp.py --reference_path work/cell_graphs --graph_path work/random_plans --save_path gene_sim_pathway/data/cell_signature_graphs_state_ablation --benchmark_path work/response --signature_utils assets/SIGnature/src/SIGnature/utils.py --datasets GSE223779 GSE140440 GSE163836 GSE202234 GSE223779_TS485T
python gene_sim_pathway/code/state_signature_ablation/verify_random_state_signatures_exp.py --reference_path work/cell_graphs --graph_path work/random_plans --open_path gene_sim_pathway/data/cell_signature_graphs_state_ablation
python model/run.py random-signature --open_path work/response --stage run
```

## External cohort workflows

The `external`, `alk-transfer` and `clinical-transfer` runners take the cohort module's prepared counts directory via `--open_path work/counts`. They read metadata from `work/cohort_metadata`, or the directory set by `CELL_XAI_COHORT_METADATA`. Use the fixed split JSONs for all external labels. For an individual GATv2/GINE source–target run, use the [frozen transfer example](../README.md#frozen-transfer).

## Grouped preparation

Prepare the selected task with the cohort module, using `work/counts` and `work/cohort_metadata`. Build the common graph resources under `work/cell_graphs`. Place SCimilarity under `assets/scimilarity` and use the SIGnature source under `assets/SIGnature`. Set `CELL_XAI_ATTR_PYTHON` and `CELL_XAI_SE_PYTHON` when attribution and SE use separate environments.

```sh
python model/run.py grouped --open_path work/counts --task GSE165897_pre_EOC --stage attribution
python model/run.py grouped --open_path work/counts --task GSE165897_pre_EOC --stage index
python model/run.py grouped --open_path work/counts --task GSE165897_pre_EOC --stage split_profile
python model/run.py grouped --open_path work/counts --task GSE165897_pre_EOC --stage graphs
python model/run.py grouped --open_path work/counts --task GSE165897_pre_EOC --stage train --models state_15b
```

For the native SE baseline, install [Arc STATE](https://github.com/ArcInstitute/state) and the dependencies in `env/state/requirements.txt`, place SE-600M at `assets/SE-600M`, then run the `state` and `cohorts` stages before training `native_state_embedding`.

[Cumulative attribution workflow](cumulative-signatures.md)

## Frozen protein node features

Map each protein representation to the graph vocabulary using the same HGNC snapshot as graph construction. Compare the three matching graph and feature pairs.

```sh
python model/response_benchmark/code/prepare/prepare_gene_embeddings_exp.py --graph_path work/cell_graphs --protein_path assets/proteins --hgnc_path assets/hgnc_complete_set.txt
python model/run.py gene-embedding --stage preflight --datasets GSE140440
python model/run.py gene-embedding --stage train --datasets GSE140440
```

The preparation writes gene-order matrices and coverage tables under `model/response_benchmark/data/gene_embeddings`. Use `CELL_XAI_GENE_FEATURES_ROOT` for a different feature directory and `CELL_XAI_BENCHMARK_ROOT` for source cohort manifests from a different run.
