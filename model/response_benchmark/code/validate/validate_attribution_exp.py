"""Validate matched-cell IG, held-out membership and saved prediction consistency."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from benchmark_paths import BENCH_ROOT
from run_attribution_exp import jobs, cohort_inputs, select_cells, checkpoint_path, RELATIONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--open_path", type=Path, default=BENCH_ROOT / "output/attribution")
    a = p.parse_args();records = []
    for job in jobs():
        expected, _ = cohort_inputs(job["cohort"], job["dataset"])
        match = None
        for model in RELATIONS:
            folder = a.open_path / job["dataset"] / job["fold"] / model
            parameters = json.loads((folder / "parameters.json").read_text())
            selected = pd.read_csv(folder / "cells.tsv", sep="\t", keep_default_na=False)
            actual_selection = select_cells(expected, job, parameters["cells_per_group"], parameters["seed"])
            assert actual_selection.cell_id.tolist() == selected.cell_id.tolist()
            assert actual_selection.y.tolist() == selected.y.tolist()
            if match is None:match = selected.cell_id.tolist()
            assert match == selected.cell_id.tolist()
            checkpoint = checkpoint_path(job, model)
            if job["fold"] == "infer":
                base = BENCH_ROOT / "output" if model == "state_15b" else BENCH_ROOT / "output/transfer" / model
                predictions = base / "transfer" / ("GSE223779_to_" + job["dataset"]) / "gat_gine/predictions.tsv"
            else:predictions = checkpoint.parent / "predictions.tsv"
            saved = pd.read_csv(predictions, sep="\t", float_precision="round_trip").set_index("cell_id")
            max_error, failures = 0., 0
            for i, row in enumerate(selected.itertuples()):
                with np.load(folder / "cells" / f"{i:05d}.npz", allow_pickle=False) as z:
                    assert str(z["cell_id"]) == row.cell_id
                    assert saved.loc[row.cell_id, "y"] == row.y
                    for diagnostic in json.loads(str(z["diagnostics"])):
                        values = z[diagnostic["baseline"] + "_signed_ig"]
                        assert len(values) == len(z["gene_id"]) and np.isfinite(values).all()
                        assert np.isclose(values.sum(), diagnostic["margin"] - diagnostic["baseline_margin"] + diagnostic["captum_delta"], atol=.001, rtol=.0001)
                        error = abs(diagnostic["margin"] - saved.loc[row.cell_id, "margin_resistant_minus_sensitive"])
                        max_error = max(max_error, float(error));assert error < .0001
                        failures += int(not diagnostic["converged"])
            for label in [0, 1]:
                with np.load(folder / f"gnn_class_{label}.npz", allow_pickle=False) as z:
                    assert str(z["cell_id"]) in selected.loc[selected.y.eq(label), "cell_id"].tolist()
                    assert z["edge_index"].shape[1] == len(z["edge_mask"])
                    assert np.isfinite(z["edge_mask"]).all() and ((z["edge_mask"] >= 0) & (z["edge_mask"] <= 1)).all()
            records.append(dict(dataset=job["dataset"], fold=job["fold"], model=model,
                                n_cells=len(selected), n_nonconverged_baselines=failures,
                                max_saved_margin_error=max_error, status="pass"))
    (a.open_path / "summary").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(a.open_path / "summary/validation.tsv", sep="\t", index=False)
    print("Validated", len(records), "model/fold jobs and", sum(r["n_cells"] for r in records), "model-cell records")


if __name__ == "__main__":main()
