"""Shared model-code locations and categorized module search paths."""
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parent
BENCH_ROOT = CODE_ROOT.parent
PROJECT_ROOT = CODE_ROOT.parents[2]
CATEGORIES = ("core", "explain", "prepare", "train", "transfer", "ablation", "report", "validate", "pipeline", "mass_signature")
for category in CATEGORIES:
    path = CODE_ROOT / category
    if path.is_dir() and str(path) not in sys.path:
        sys.path.append(str(path))


def script_path(name):
    """Find the single categorized source file for a script or resource."""
    if Path(name).name != name:
        raise ValueError("Expected a filename: " + name)
    found = [CODE_ROOT / category / name for category in CATEGORIES
             if (CODE_ROOT / category / name).is_file()]
    if not found and (CODE_ROOT / name).is_file():
        found = [CODE_ROOT / name]
    if len(found) != 1:
        raise FileNotFoundError("Expected one implementation for " + name)
    return found[0]
