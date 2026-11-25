# Compare cell encoder attribution

Run SCimilarity, CZI scVI and local scVI on the same ordered count matrix. Write all three IG outputs to the same `work/attributions/GSE*` cohort directory.

```sh
python -m pip install numpy scipy matplotlib matplotlib-venn
python signature_attr_run/code/compare_importance.py --open_path work/attributions --save_path work/encoder_comparison --top_k 10 50 100
```

`spearman_dataset.tsv` compares the mean per-cell importance on pairwise common genes. The active comparison retains genes with a positive mean in either encoder. `matrix_qc.tsv` records matrix sizes and row sums. The overlap outputs use either the three-way common gene universe or each encoder's native vocabulary.

The historical comparison used GSE115978, GSE140440, GSE143363, GSE149383, GSE186960 and GSE223779. Use the original count subsets for exact replication. GSE149383 here is the original benchmark subset, with a different cell set from the grouped PC9 response task. The cohort pipeline covers the response benchmarks. Contact the authors for the exact encoder-comparison count subsets when they cannot be recovered from those inputs.
