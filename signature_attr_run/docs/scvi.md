# scVI attribution

Install SIGnature using its upstream instructions. Use a separate Python 3.12 environment with PyTorch 2.5.1 and install `signature_attr_run/requirements-scvi.txt` for these runs.

```sh
python signature_attr_run/code/run_attribution.py --model scvi_czi --h5ad work/counts/GSE140440/counts.h5ad --scvi-model-dir assets/scvi/2023-12-15-scvi-homo-sapiens/scvi.model --ensembl_gene_file assets/scvi/ensembl_gene_symbols.txt --method ig --batch-size 100 --out-dir work/attributions/GSE140440
python signature_attr_run/code/run_attribution.py --model scvi_local --h5ad work/counts/GSE140440/counts.h5ad --scvi-hvg 2000 --scvi-epochs 100 --method ig --batch-size 100 --out-dir work/attributions/GSE140440
```

CZI attribution uses the fixed checkpoint vocabulary. Local scVI selects Seurat v3 HVGs from the count layer of the supplied cohort and fits the encoder before attribution. Full runs use every eligible cell. Sparse outputs are cell-by-gene, with explicit row and column identifiers. These attributions explain the encoder representation and are separate from classifier attribution.

Retain the checkpoint gene-overlap diagnostics when comparing cohorts. Low overlap indicates that many checkpoint features are absent from the input.
