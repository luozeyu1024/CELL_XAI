"""Read authoritative cell-ID/label JSONs without generating new partitions."""
import json
import os
from pathlib import Path
import pandas as pd

DEFAULT_ROOT = Path(__file__).resolve().parents[4] / "dataset_util/datasets_download_preprocess/26_11_20/splits"
PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROLES = {"train": "train", "val": "validation", "test": "test"}

def cohort_metadata_root():
    return Path(os.environ.get("CELL_XAI_COHORT_METADATA", PROJECT_ROOT / "work/cohort_metadata")).resolve()


def prostate_cell_lines():
    resource = DEFAULT_ROOT.parent / "resources/sample_conditions.tsv"
    metadata = pd.read_csv(resource, sep="\t", dtype=str, keep_default_na=False)
    return metadata.loc[metadata.dataset.eq("GSE140440")].set_index("sample_id", verify_integrity=True)["cell line"]


def split_root():
    root = Path(os.environ.get("CELL_XAI_SPLITS_ROOT", DEFAULT_ROOT)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("Split JSON directory not found: " + str(root))
    return root


def fold_from_path(path):
    return next((part for part in reversed(Path(path).parts)
                 if part.startswith("fold_") and part[5:].isdigit()), "single")


def read_records(path, role):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty list: " + str(path))
    for row in records:
        if not isinstance(row, dict) or set(row) != {"cell_id", "label"}:
            raise ValueError("Expected cell_id and label only: " + str(path))
        if not isinstance(row["cell_id"], str) or not row["cell_id"]:
            raise ValueError("Invalid cell ID: " + str(path))
        if type(row["label"]) is not int or row["label"] not in (-1, 0, 1):
            raise ValueError("Invalid label: " + str(path))
        if (role == "excluded_unknown_response") != (row["label"] == -1):
            raise ValueError("Unknown/known label in the wrong file: " + str(path))
    frame = pd.DataFrame(records).rename(columns={"label": "y"})
    if frame.cell_id.duplicated().any():
        raise ValueError("Duplicate cell IDs: " + str(path))
    frame["split"] = role
    return frame


def assignments(dataset, fold="single", mode="auto", scope=None):
    folder = split_root() / dataset
    if fold != "single":
        folder = folder / fold
    if mode not in ("auto", "source", "infer"):
        raise ValueError("Invalid split mode: " + mode)
    external = mode == "infer" or (mode == "auto" and (folder / "infer.json").exists())
    parts = []
    if external and (folder / "infer.json").exists():
        selected = folder / scope if scope else folder
        parts.append(read_records(selected / "infer.json", "external"))
        if not scope and (folder / "unknown.json").exists():
            parts.append(read_records(folder / "unknown.json", "excluded_unknown_response"))
    else:
        if scope:
            raise ValueError("Subgroup inference JSON not found: " + str(folder / scope))
        for filename, role in ROLES.items():
            parts.append(read_records(folder / (filename + ".json"), role))
        if external:
            for part in parts:
                part["split"] = "external"
    frame = pd.concat(parts, ignore_index=True)
    if frame.cell_id.duplicated().any():
        raise ValueError("Cell appears in multiple split files: " + str(folder))
    return frame


def apply_splits(cells, dataset, fold="single", mode="auto", scope=None):
    """Keep matrix row order; select and label cells strictly by registered JSON IDs."""
    frame = cells.copy()
    if "cell_id" not in frame or frame.cell_id.duplicated().any():
        raise ValueError("Missing or duplicate cell_id column: " + dataset)
    frame["cell_id"] = frame.cell_id.astype(str)
    assigned = assignments(dataset, fold, mode, scope).set_index("cell_id")
    missing = assigned.index.difference(frame.cell_id)
    if len(missing):
        raise ValueError(f"{dataset}: {len(missing)} JSON cell IDs missing from data: {missing[:5].tolist()}")
    if scope:
        frame = frame[frame.cell_id.isin(assigned.index)].copy()
    extra = ~frame.cell_id.isin(assigned.index)
    if extra.any():
        raise ValueError(f"{dataset}: {int(extra.sum())} data cells absent from split JSONs")
    labels = frame.cell_id.map(assigned.y).astype(int)
    if "y" in frame:
        previous = pd.to_numeric(frame.y, errors="raise").astype(int)
        conflict = previous.ne(labels)
        if conflict.any():
            raise ValueError(f"{dataset}: label conflicts for {frame.loc[conflict, 'cell_id'].head().tolist()}")
    frame["y"] = labels
    frame["split"] = frame.cell_id.map(assigned.split)
    return frame


def load_cells(path, dataset, mode="auto", fold=None):
    path = Path(path)
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    return apply_splits(frame, dataset, fold or fold_from_path(path), mode)


def source_folds(dataset):
    folder = split_root() / dataset
    if (folder / "train.json").is_file():
        return ["single"]
    folds = sorted((p.name for p in folder.glob("fold_*") if p.is_dir()),
                   key=lambda value: int(value.split("_")[1]))
    if not folds:
        raise FileNotFoundError("No registered source folds for " + dataset)
    return folds
