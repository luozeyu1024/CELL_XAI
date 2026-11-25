# Cell encoder attribution

Run per-cell gene attribution with the [SIGnature environment and model files](https://genentech.github.io/SIGnature/notebooks/2-GeneratingAttributions.html). Start with raw counts from the cohort module and the SCimilarity checkpoint for graph construction.

See [download and placement instructions](../docs/data-and-model-resources.md#cell-encoder-files) for the exact checkpoint filenames.

```sh
python signature_attr_run/code/run_attribution.py --model scimilarity --h5ad work/counts/GSE140440/counts.h5ad --model-dir assets/scimilarity --method ig --out-dir work/attributions/GSE140440
```

Each cohort folder contains a sparse cell-by-gene attribution matrix, ordered gene names and a manifest containing cell IDs and input coordinates. Keep the original H5AD available at the path recorded in the manifest.

Use `scvi_czi` with `--scvi-model-dir` and `--ensembl_gene_file` for the pretrained CZI checkpoint. Use `scvi_local` for local training with Seurat v3 HVGs. The graph workflow consumes SCimilarity IG.

[scVI model setup and commands](docs/scvi.md)

[Compare gene importance across encoders](docs/compare-importance.md).

For a CPU smoke run, append `--device cpu --max-cells 2 --batch-size 2` and use a separate output directory. Full graph preparation requires attribution for all registered cohort cells.
