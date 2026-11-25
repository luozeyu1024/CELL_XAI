# Attribution distributions

Compare SIGnature and three frozen graph classifiers on each method's own Top 100 or a shared SIGnature Top 100. Set `CELL_XAI_BENCHMARK_ROOT` to the trained benchmark directory if it differs from `model/response_benchmark`.

The pipeline reads SCimilarity manifests from `work/attributions`. Set `CELL_XAI_ATTRIBUTION_ROOT` to reuse a different attribution directory.

```sh
python analysis/attribution_distribution/code/compute_full_ig_exp.py --stage base --datasets GSE140440
python analysis/attribution_distribution/code/summarize_exp.py --stage base --datasets GSE140440
python analysis/attribution_distribution/code/plot_exp.py --stage base --datasets GSE140440
python analysis/attribution_distribution/code/validate_exp.py --stage base --datasets GSE140440
```

Compute with `--stage target` once the external cohorts are prepared, then summarize and plot with `--stage all`. Each PDF has a companion JSON with gene means, quantiles, confidence intervals and reference coverage. All four methods are compared on common cell nodes, with absolute attribution normalized to 1,000 per cell. Mean confidence intervals describe the selected gene values.

The external ovarian cohort must have a grouped graph package at the corresponding source fold's K. Use `CELL_XAI_GROUPED_GRAPHS_ROOT` if these packages are stored elsewhere.

The GSE223779 source is evaluated on external TS485T organoids and three cell-line cohorts. Each pretreatment EOC source fold is evaluated on GSE154600 epithelial cells. The comparison keeps these target settings distinct and excludes unknown response labels.
