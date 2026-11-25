"""Check JSON partitions against cohort rows and optionally saved prediction tables."""
import argparse
from pathlib import Path
import sys
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import benchmark_paths
from split_registry import assignments, apply_splits, load_cells, source_folds, split_root, fold_from_path, cohort_metadata_root

BENCH = CODE.parent
PROJECT = BENCH.parent.parent


def validate_cohorts():
    checked = 0
    for folder in sorted(split_root().iterdir()):
        if not folder.is_dir():
            continue
        dataset = folder.name
        if (folder / "infer.json").is_file():
            if dataset == "GSE154600_EPI_external":
                path = PROJECT / "gene_sim_pathway/data/cell_signature_graphs_grouped/K768" / dataset / "cells.tsv"
                cells = pd.read_csv(path, sep="\t", keep_default_na=False)
                meta = pd.read_csv(cohort_metadata_root() / dataset / "counts_labels.tsv",
                                   sep="\t", keep_default_na=False).set_index("Cell_ID")
                cells["y"] = cells.cell_id.map(meta.label).map({"sensitive": 0, "resistant": 1}).fillna(-1).astype(int)
                apply_splits(cells, dataset, mode="infer")
            else:
                load_cells(BENCH / "data/cohorts" / dataset / "cells.tsv", dataset, mode="infer")
            for subgroup in folder.iterdir():
                if subgroup.is_dir() and (subgroup / "infer.json").is_file():
                    sub = assignments(dataset, mode="infer", scope=subgroup.name)
                    full = assignments(dataset, mode="infer").set_index("cell_id")
                    if not sub.cell_id.isin(full.index).all():
                        raise ValueError("Subgroup IDs outside main inference list")
                    if sub.y.tolist() != full.loc[sub.cell_id, "y"].tolist():
                        raise ValueError("Subgroup labels differ")
            checked += 1
        else:
            for fold in source_folds(dataset):
                root = BENCH if fold == "single" else BENCH / "data/grouped" / dataset / fold
                cells = load_cells(root / "data/cohorts" / dataset / "cells.tsv", dataset, mode="source")
                original = pd.read_csv(root / "data/cohorts" / dataset / "cells.tsv", sep="\t", keep_default_na=False)
                if cells["split"].tolist() != original["split"].tolist():
                    raise ValueError("Existing cohort split differs from JSON: " + str(root))
                checked += 1
    print("Validated cohort/fold definitions:", checked)


def validate_predictions():
    checked = 0
    for path in sorted((BENCH / "output").rglob("predictions.tsv")):
        if "within" not in path.parts:
            continue
        dataset = path.parent.parent.name
        if not (split_root() / dataset).is_dir():
            continue
        if "transfer" in path.parts:
            continue
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        normalized = apply_splits(frame, dataset, fold_from_path(path), mode="source")
        if normalized["split"].tolist() != frame["split"].tolist():
            raise ValueError("Saved predictions use different JSON splits: " + str(path))
        checked += 1
    # Source checkpoints under transfer/ are still source-trained within/ results.
    for path in sorted((BENCH / "output/transfer").rglob("predictions.tsv")):
        if path.parent.parent.name != "GSE223779" or path.parent.parent.parent.name != "within":
            continue
        frame = pd.read_csv(path, sep="\t", keep_default_na=False)
        checked_frame = apply_splits(frame, "GSE223779", mode="source")
        if frame["split"].tolist() != checked_frame["split"].tolist():
            raise ValueError("Saved source split differs: " + str(path))
        checked += 1
    print("Validated saved source prediction tables:", checked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check_results", action="store_true")
    args = parser.parse_args()
    validate_cohorts()
    if args.check_results:
        validate_predictions()


if __name__ == "__main__":
    main()
