# Expression-weighted cell embeddings

Compute cell representations from expression and a gene-to-vector dictionary using the GenePT-style mapper.

Install the mapper dependencies with `python -m pip install -r PLM_cell_emb/requirements.txt`.

```sh
python PLM_cell_emb/code/gene_to_protein_embedding_mapper_Genept_v.py --h5ad_file work/counts/GSE140440/counts.h5ad --genept_pickle assets/gene_vectors.pkl --save_path work/cell_embeddings/GSE140440
```

Use a trusted pickle keyed by gene symbol. The mapper aligns genes, applies its configured expression normalization and writes `genept_cell_embedding.npy`. The other mapper entry points accept protein vectors with an HGNC mapping, or a STATE gene/protein embedding dictionary. Each entry point exposes its input options through `--help`.
