# Model input contracts

## Cell graphs

Each graph package has a common `gene_vocab.tsv` and one CSR matrix under `global_graphs/` per relation. Each cohort has `cells.tsv`, `parameters.json` and `signatures.npz`. The NPZ stores offsets `ptr`, ordered node indices `gene_id` and node `features`. The classifier reads expression from the first feature column. SIGnature IG determines which genes enter the graph.

The preparation command joins graph cells to the registered split IDs and saves their graph rows in `data/cohorts/<dataset>/cells.tsv`. Graph rows are preserved even when unregistered cells are omitted. Label −1 identifies unknown response and is excluded from binary metrics. For archived SE outputs with 2,058 columns, the preparation command keeps the first 2,048 after matching cell IDs.

## Fit and evaluate

Fit preprocessing and model parameters on the source training cells. Use validation cells for checkpoint selection and preserve test and external IDs for evaluation. Transfer uses a source checkpoint and the same ordered gene vocabulary. Baseline type, node selection and graph relation are independent experimental settings.

Default graph settings are 64 hidden features, two layers, four attention heads, 16 gene identity features and dropout 0.2. Training defaults are AdamW, learning rate 0.001, weight decay 0.0001, seed 42 and early stopping after 15 validation epochs without improvement.
