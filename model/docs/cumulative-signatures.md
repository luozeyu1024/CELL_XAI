# Cumulative attribution signatures

Use the original per-cell attribution sum as the denominator. For each threshold (0.50, 0.60, 0.70 and 0.80), take the shortest ordered prefix of positive eligible genes that reaches it. Keep all positive eligible genes and flag the cell if the threshold is unreachable.

Start with the graph package and SCimilarity output for each cohort. The first step writes candidate arrays and per-cell topology statistics. The second retrieves expression for newly included nodes using the original count matrix and SIGnature preprocessing.

```sh
python model/run.py prepare --open_path work/cell_graphs --save_path model/response_benchmark --dataset GSE140440
python model/response_benchmark/code/mass_signature/profile_cumulative_graphs_exp.py --graph_path work/cell_graphs --attribution_path work/attributions --datasets GSE140440
python model/response_benchmark/code/mass_signature/prepare_cumulative_expression_exp.py --datasets GSE140440 --counts_root work/counts
python model/response_benchmark/code/mass_signature/validate_cumulative_graphs_exp.py --datasets GSE140440
python model/response_benchmark/code/mass_signature/summarize_cumulative_profiles_exp.py
python model/response_benchmark/code/mass_signature/benchmark_cumulative_gpu_exp.py --datasets GSE140440 --batch_size 32
```

Use the [recorded threshold choices](../config/cumulative-thresholds.json) to reproduce the source designs. The selector checks source-training topology and GPU feasibility before freezing each policy. A separate choices file can specify `DATASET__FOLD` keys to override individual folds in a new run.

```sh
python model/response_benchmark/code/mass_signature/select_cumulative_policies_exp.py --choices model/config/cumulative-thresholds.json --datasets GSE140440
python model/run.py cumulative --stage train --datasets GSE140440
```

Candidate packages live under `gene_sim_pathway/data/cumulative_signature_graphs`; topology, GPU profiles and selected policies are under `work/cumulative_signature/source_data`. Prepare target packages before running frozen transfer. Each target uses its source fold's threshold and checkpoint.

Set `CELL_XAI_BENCHMARK_ROOT` to read existing source cohort manifests from another run directory.

## Compare completed models

After generating the fixed-K, cumulative and frozen-protein results, run the reports from the repository root. These commands summarize saved predictions and checkpoints.

`model/run.py transfer`, `no-edges` and `random-signature` each provide a `--stage report` command for their fixed-K outputs. Complete the external, ALK and ovarian workflows in the [model guide](advanced-workflows.md#external-cohort-workflows) before merging their metrics.

```sh
python model/response_benchmark/code/report/report_mechanism_exp.py --open_path model/response_benchmark
python model/response_benchmark/code/report/report_pc9_state_exp.py
python model/response_benchmark/code/report/stratify_source_predictions_exp.py --save_path work/transfer_analysis/data/GSE140440_frozen_cell_line_metrics.tsv
python model/response_benchmark/code/report/integrate_reassessed_exp.py --open_path model/response_benchmark/output/summary --strata_path work/transfer_analysis/data/GSE140440_frozen_cell_line_metrics.tsv --save_path work/transfer_analysis/data/reassessed
python model/response_benchmark/code/mass_signature/report_cumulative_benchmark_exp.py
python model/response_benchmark/code/mass_signature/report_cumulative_edge_ablation_exp.py --stage all
python model/response_benchmark/code/report/report_gene_embedding_exp.py --variant state_graph__state_15b
python model/response_benchmark/code/mass_signature/compare_all_response_models_exp.py
```

The fixed-K summaries require the complete grouped, graph transfer and ablation outputs. GSE140440 frozen predictions are split into DU145 and PC3 using the bundled GEO sample annotations. The combined comparison writes `all_models_formal_comparison.tsv`, `all_model_definitions.tsv` and `all_models_training_comparison.tsv` under `work/cumulative_signature/source_data`. The formal comparison matches models by evaluation scope. The training comparison retains the original source folds.
