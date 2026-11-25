# Cohort preparation

Download GEO data, reconstruct raw count matrices and recover fixed train, validation, test and inference cell assignments.

```sh
python -m pip install -r dataset_util/datasets_download_preprocess/26_11_20/requirements.txt
python dataset_util/datasets_download_preprocess/26_11_20/code/reproduce_datasets_exp.py --stage download --datasets GSE140440 --open_path work/raw
python dataset_util/datasets_download_preprocess/26_11_20/code/reproduce_datasets_exp.py --stage preprocess --datasets GSE140440 --open_path work/raw --save_path work/counts --metadata_path work/cohort_metadata
python dataset_util/datasets_download_preprocess/26_11_20/code/reproduce_datasets_exp.py --stage verify --datasets GSE140440 --save_path work/counts --metadata_path work/cohort_metadata
```

Each prepared cohort contains a cells-by-genes `counts.h5ad`. [Cohort definitions and download URLs](datasets.json) identify the source files. [Split JSONs](splits/) contain cell IDs and labels. GSE163836 requires R and Matrix. Source data and annotations retain the terms of their original providers.

`resources/cohort_axes.json.gz` stores the exact ordered cell and gene IDs used by preprocessing and verification.

`--check_only` verifies reconstruction without saving a count matrix. Add `--reference_path` to compare count values and ordered identifiers to an existing cohort directory.

If the download server rejects a request, retrieve the archive from the accession's GEO supplementary files page and place it under `work/raw` at the relative `path` specified in `datasets.json`. Run preprocessing only after the full files are present. Files ending in `.part` are incomplete downloads.
