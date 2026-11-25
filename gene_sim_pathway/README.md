# Protein and cell graphs

Build exact cosine protein graphs, map them to genes, then select per-cell nodes using SCimilarity IG. The graph package stores four sparse gene graphs and ordered per-cell node lists. The model loader extracts each cell's induced subgraph.

Place the aligned protein NPZ bundles under `assets/proteins/{ESM2_650M,Qwen,STATE_ESM2_15B}/human_swissprot_aligned.npz`.

```sh
python gene_sim_pathway/code/emb_KNN_graph_G/build_aligned_embedding_knn_exp.py --open_path assets/proteins --save_path assets/proteins/emb_KNN_graph --model esm2_650m qwen state_15b --n_neighbors 30
python gene_sim_pathway/code/emb_KNN_graph_G/build_cell_signature_graphs_exp.py --open_path assets/proteins --save_path work/cell_graphs --attribution_path work/attributions --hgnc_path assets/hgnc_complete_set.txt --ppi_path assets/format_h_sapiens.csv --signature_utils assets/SIGnature/src/SIGnature/utils.py --datasets GSE140440 --policy_path gene_sim_pathway/config/fixed-k.json --ppi_threshold 0.5
```

The example uses the recorded fixed K for GSE140440. To run a fixed-K comparison, replace the policy argument with `--top_k 512`. Pass the graph package to the [response modeling workflow](../model/README.md) for training.

[Graph inputs and file formats](docs/graph-inputs.md)
