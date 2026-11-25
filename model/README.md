# Response models

Train GATv2/GINE graph classifiers or SE embedding baselines, evaluate frozen transfer and attribute predictions to genes and edges. Training and inference read the bundled [split JSONs](../dataset_util/datasets_download_preprocess/26_11_20/splits/).

Use Python 3.11 and install `python -m pip install -r model/requirements.txt`. For CPU execution append `--device cpu` to the training command.

```sh
python model/run.py prepare --open_path work/cell_graphs --save_path work/response --dataset GSE140440
python model/run.py train --open_path work/response --stage train --datasets GSE140440 --models gat_gine --relation state_15b
python model/run.py explain --help
```

The implementation is grouped under [response_benchmark/code](response_benchmark/code/) by preparation, training, transfer, ablation, explanation and reporting. Cumulative signature experiments live under `mass_signature/`.

The preparation step joins registered IDs to graph rows. For the SE baseline, supply `--state_h5ad` and the appropriate `--state_key` during preparation, then train with `--models state_mlp`.

[Input contracts and model settings](docs/input-contracts.md)

## Frozen transfer

Train the source model and prepare the target against the same graph vocabulary. The example transfers from a source organoid line to an external organoid line.

Build both organoid graph cohorts with `--datasets GSE223779 GSE223779_TS485T --policy_path gene_sim_pathway/config/fixed-k.json` so both use the source K of 1024.

```sh
python model/run.py prepare --open_path work/cell_graphs --save_path work/response --dataset GSE223779
python model/run.py train --open_path work/response --stage train --datasets GSE223779 --models gat_gine --relation state_15b
python model/run.py prepare --open_path work/cell_graphs --save_path work/response --dataset GSE223779_TS485T
python model/run.py train --open_path work/response --stage transfer --source_dataset GSE223779 --datasets GSE223779_TS485T --models gat_gine --relation state_15b
```

The runner writes checkpoints and predictions under `work/response/output/`.

[Grouped experiments, ablations and explanations](docs/advanced-workflows.md)
