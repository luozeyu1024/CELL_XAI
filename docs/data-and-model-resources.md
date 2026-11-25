# Data and model resources

| Resource | Source or preparation |
| --- | --- |
| Cohort counts | Download the GEO input files using the [cohort workflow](../dataset_util/datasets_download_preprocess/26_11_20/README.md). The repository includes the cohort catalog, compressed ordered identifiers, annotation resources and fixed split JSONs. |
| Cell encoder weights | Download `model_files.tar.gz` from [SIGnature Model Files, Zenodo v1](https://zenodo.org/records/17903196). It contains SCimilarity weights and the scVI gene mapping. The CZI scVI checkpoint is a separate download. |
| Graph resources | Obtain the HGNC snapshot and scNET interaction file, then prepare ESM2-650M, Qwen and STATE ESM2-15B bundles using the [graph input specification](../gene_sim_pathway/docs/graph-inputs.md). |
| Native SE baseline | Obtain [SE-600M](https://huggingface.co/arcinstitute/SE-600M) and follow the [model guide](../model/docs/advanced-workflows.md). |
| Response checkpoints and attribution results | Generate them with the [training and transfer workflow](../model/README.md) and [attribution analysis](../analysis/attribution_distribution/README.md). |

## Cell encoder files

Extract the Zenodo archive and place its `scimilarity/` and `scvi/` directories under `assets/`. For scVI, download the [human Census checkpoint dated 2023-12-15](https://cellxgene-contrib-public.s3.us-west-2.amazonaws.com/models/scvi/2023-12-15/homo_sapiens/model.pt) into `assets/scvi/2023-12-15-scvi-homo-sapiens/scvi.model/model.pt`. The gene mapping from the Zenodo archive belongs at `assets/scvi/ensembl_gene_symbols.txt`. Check the [CELLxGENE model catalog](https://cellxgene.cziscience.com/census-models) if the direct link changes.

For the native SE baseline, download `config.yaml`, `model.safetensors` and `protein_embeddings.pt` from the [SE-600M files](https://huggingface.co/arcinstitute/SE-600M/tree/main) into `assets/SE-600M/`.

## Protein graph inputs

Use the [scNET interaction file at the recorded revision](https://github.com/madilabcode/scNET/blob/e0ef9c7a76d566afc80a7c8cfb88a11afbf60310/scNET/Data/format_h_sapiens.csv) and the [bundled HGNC snapshot](../gene_sim_pathway/resources/hgnc_complete_set.txt.gz). The graph guide specifies the required fields and paths. The [ESM upstream repository](https://github.com/Biohub/esm-archive#extract-per-residue-representations) provides the 650M sequence model.

The [graph input guide](../gene_sim_pathway/docs/graph-inputs.md) reconstructs the aligned STATE bundle from the public SE-600M protein asset and the included reference data. It also provides the frozen UniProt sequences and GO annotations for recomputing ESM2-650M and Qwen vectors.

**Historical vector files.** Contact the authors for the original aligned protein NPZ files if you need to reproduce the historical graph inputs byte for byte. Specify the representation name and the filename listed below. For STATE-15B, also request the accession map if using the archived export. Hosted Qwen outputs can change, so use the author-provided Qwen NPZ for an exact historical run.

The SE-600M `protein_embeddings.pt` used for reconstruction is available at [revision 0fccde9](https://huggingface.co/arcinstitute/SE-600M/blob/0fccde9e9c43a894a53c1e1c5f192ec4dd0fb033/protein_embeddings.pt).

For an exact run, compare supplied inputs to the reference file hashes below. Use `sha256sum` on Linux or `Get-FileHash -Algorithm SHA256` in PowerShell.

| Input relative to `assets/` | Reference SHA-256 |
| --- | --- |
| `hgnc_complete_set.txt` | `1f93c0aaa8ef1b65ecdf3b956c457d8389173f1ce0e2f3d873c1ef76825dc1fd` |
| `format_h_sapiens.csv` | `6818109d55256211574849256204fc3dc4ff5e15c0dc49b3f20adb43d5d5ac45` |
| `proteins/ESM2_650M/human_swissprot_aligned.npz` | `f1f018a794841eac37a06c5ad1a967ae2d217a960fb3cb6fab50bff48663f198` |
| `proteins/Qwen/human_swissprot_aligned.npz` | `3cd17c32c44cd91c4e9ef456484d9d14bc3fa02e2e7a68c56406250083b56efa` |
| `proteins/STATE_ESM2_15B/human_swissprot_aligned.npz` | `4d8ba2110730a16df6078b24804579abc0b8c5377874d7b200557c743f717686` |
| `proteins/STATE_ESM2_15B/protein_gene_mapping.tsv` | `d82687610d5266628d71337ee80c7c3f62b5875f5890cd138514dd08c4138141` |

Keep upstream assets under `assets/` and run outputs under `work/` or the output paths shown in the module commands. These locations are excluded from version control.
