# Cell XAI

Analyze how gene expression and protein-derived graph structure contribute to single-cell response predictions. The workflow links reproducible cohort definitions, per-cell gene attribution, graph construction, response modeling and model explanations.

## Workflow

| Step | Input | Output |
| --- | --- | --- |
| [1. Prepare cohorts](dataset_util/datasets_download_preprocess/26_11_20/README.md) | GEO downloads and cohort definitions | Count matrices, cell labels and fixed splits |
| [2. Compute gene attributions](signature_attr_run/README.md) | Counts and a cell encoder | Per-cell gene importance with ordered identifiers |
| [3. Build graphs](gene_sim_pathway/README.md) | Protein embeddings, interaction edges and gene importance | Shared sparse graphs and per-cell node lists |
| [4. Train, transfer and explain](model/README.md) | Graph packages and fixed split JSONs | Checkpoints, predictions, ablations and attribution summaries |
| [5. Compare attribution distributions](analysis/attribution_distribution/README.md) | Frozen classifiers and aligned node importance | Paired PDF and statistical JSON |
| [Optional. Build cell embeddings](PLM_cell_emb/README.md) | Expression and aligned gene vectors | Expression-weighted cell representations |

Start with the single-cohort commands in each module. Use the same cell IDs and gene symbol mapping across stages. The module pages specify the Python environment and expected input files.

See [Data and model resources](docs/data-and-model-resources.md) for the included metadata, upstream downloads and external assets required by each workflow.
