# Graph inputs and file formats

| Input | Required fields | Source |
| --- | --- | --- |
| Aligned protein NPZ | `ids`, `gene_names`, `embeddings` in the same row order | Human Swiss-Prot proteins with ESM2-650M sequence vectors, Qwen protein-annotation text vectors or STATE ESM2-15B gene vectors |
| STATE accession map | `accession`, `hgnc_symbol` columns in `STATE_ESM2_15B/protein_gene_mapping.tsv` | Accession-to-gene mapping for the STATE embedding bundle |
| HGNC mapping | Approved symbols, aliases and UniProt accessions | [HGNC complete set archive](https://www.genenames.org/download/archive/) |
| Interaction edges | `g1_symbol`, `g2_symbol`, `conn` | [scNET `format_h_sapiens.csv`](https://raw.githubusercontent.com/madilabcode/scNET/e0ef9c7a76d566afc80a7c8cfb88a11afbf60310/scNET/Data/format_h_sapiens.csv). Save as `assets/format_h_sapiens.csv`. |
| SCimilarity IG | Sparse attribution rows, ordered gene names and source H5AD manifest | [Cell encoder attribution](../../signature_attr_run/README.md) |

The [reviewed human UniProt inventory](../resources/human_swissprot_2025-02-28.tsv.gz) and [GO input dictionary](../resources/human_go_annotations_2025-02-28.json.gz) preserve the February 2025 snapshot used for the vectors. They contain 18,432 accessions. The inventory retains accession, gene names and sequence only. Source is the UniProt Consortium, [CC BY 4.0](https://www.uniprot.org/help/license).

To restore the original [HGNC](https://hgnc.genenames.org/about/license/) complete-set snapshot from the repository, run the following command at the project root.

```sh
python -c "import gzip, pathlib; p=pathlib.Path('assets/hgnc_complete_set.txt'); p.parent.mkdir(exist_ok=True); p.write_bytes(gzip.decompress(pathlib.Path('gene_sim_pathway/resources/hgnc_complete_set.txt.gz').read_bytes()))"
```

Protein KNN uses exact cosine similarity, top 30 per protein, then symmetrizes the union. Mapped edges sharing endpoints retain the maximum weight. PPI edges use the inclusive confidence cutoff 0.5 by default. Node selection takes up to K genes with positive attribution on the supported common background. Global adjacency matrices use the ordered `gene_vocab.tsv`; cells reference its integer indices.

For candidate ranking by protein similarity, use `code/gene_target_neighbors_exp.py --help`. Ranked output separates each model's candidates, fused rank and network annotations.

## Package existing protein vectors

```sh
python gene_sim_pathway/code/pack_protein_embeddings.py --features work/esm2_650m_features.npz --inventory gene_sim_pathway/resources/human_swissprot_2025-02-28.tsv.gz --save_path assets/proteins/ESM2_650M/human_swissprot_aligned.npz
python gene_sim_pathway/code/pack_protein_embeddings.py --state_pt assets/SE-600M/protein_embeddings.pt --hgnc_path assets/hgnc_complete_set.txt --inventory gene_sim_pathway/resources/human_swissprot_2025-02-28.tsv.gz --save_path assets/proteins/STATE_ESM2_15B/human_swissprot_aligned.npz
```

The input feature NPZ pairs accession IDs with numeric vectors. The packer joins gene names and optional sequences by exact UniProt accession while preserving feature row order. Use mean-pooled sequence vectors for ESM2-650M and protein-annotation text vectors for Qwen. The STATE command reads `obs/gene_symbol` and dense `X` from the gene embedding H5AD. It retains only unambiguous approved HGNC matches and writes the accession map alongside the bundle.

For the original multi-pooling ESM2 export, pass `--esm2_matrix` with the NPY matrix, `--esm2_columns` with its feature index XLSX, and `--inventory` with its paired filtered XLSX in matrix row order. The packer selects the 1,280 mean-pooled columns. For Qwen, pass the existing annotation-vector NPZ with `--features` and the full Swiss-Prot inventory with `--inventory`.

## Recompute annotation vectors

Use the original accession-to-GO JSON or XLSX with `protein_ID` and `GO_annotation`. The encoder uses DashScope `text-embedding-v4`, 1,024 dimensions and batches of 10. Install the `openai` client and set `DASHSCOPE_API_KEY` for API execution.

```sh
python gene_sim_pathway/code/embed_protein_annotations.py --open_path gene_sim_pathway/resources/human_go_annotations_2025-02-28.json.gz --save_path work/qwen_features.npz
python gene_sim_pathway/code/pack_protein_embeddings.py --features work/qwen_features.npz --inventory gene_sim_pathway/resources/human_swissprot_2025-02-28.tsv.gz --save_path assets/proteins/Qwen/human_swissprot_aligned.npz
```

Use `--prompts_only` with a JSON output to inspect the model inputs offline. To reproduce the reported Qwen graph with the original vectors, contact the authors for the historical aligned NPZ and place it at `assets/proteins/Qwen/human_swissprot_aligned.npz`. Hosted model updates can change regenerated vectors. See the [asset reference](../../docs/data-and-model-resources.md#protein-graph-inputs) for file verification.

## Recompute ESM2 sequence vectors

Use Transformers 4.30.0 and PyTorch 2.5.1 to extract the final hidden-layer mean across residues. The sequence filter retains lengths 1 through 4,023, which selects the original 18,352 accessions. The command downloads the model from [Hugging Face](https://huggingface.co/facebook/esm2_t33_650M_UR50D) or accepts a retrieved model directory via `--model_path`.

```sh
python gene_sim_pathway/code/embed_protein_sequences.py --inventory gene_sim_pathway/resources/human_swissprot_2025-02-28.tsv.gz --save_path work/esm2_650m_features.npz --device cuda
```
