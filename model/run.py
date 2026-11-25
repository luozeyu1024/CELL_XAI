"""Run response-model experiments using the authoritative split registry."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "explain": "explain/run_attribution_exp.py",
    "attribution-report": "report/report_attribution_exp.py",
    "attribution-plot": "report/plot_attribution_exp.py",
    "prepare": "prepare/prepare_cohorts_exp.py",
    "train": "train/train_response_exp.py",
    "grouped": "train/mechanism_grouped_exp.py",
    "pc9": "train/pc9_state_grouped_exp.py",
    "gene-embedding": "train/gene_embedding_benchmark_exp.py",
    "transfer": "transfer/graph_transfer_exp.py",
    "external": "transfer/external_transfer_exp.py",
    "alk-transfer": "transfer/alk_frozen_transfer_exp.py",
    "clinical-transfer": "transfer/ovarian_frozen_transfer_exp.py",
    "no-edges": "ablation/edge_ablation_exp.py",
    "random-signature": "ablation/signature_ablation_exp.py",
    "cumulative": "mass_signature/train_cumulative_benchmark_exp.py",
    "cumulative-no-edges": "mass_signature/cumulative_edge_ablation_exp.py",
    "validate": "validate/validate_results_exp.py",
    "report": "report/report_results_exp.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits_path", type=Path,
                        default=ROOT.parent / "dataset_util/datasets_download_preprocess/26_11_20/splits")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.splits_path.is_dir():
        parser.error("Split directory not found: " + str(args.splits_path))
    env = os.environ.copy()
    env["CELL_XAI_SPLITS_ROOT"] = str(args.splits_path.resolve())
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    result = subprocess.run([sys.executable, str(ROOT / "response_benchmark/code" / COMMANDS[args.command]),
                             *arguments], env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
