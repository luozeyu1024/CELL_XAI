"""Compute ESM2-650M last-layer residue means from the reviewed inventory."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--model_path", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    inventory = pd.read_csv(args.inventory, sep="\t", dtype=str, keep_default_na=False)
    inventory = inventory.loc[inventory.Sequence.str.len().between(1, 4023)]
    if inventory.empty or not inventory.Entry.is_unique:
        parser.error("Expected eligible sequences with unique accessions")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path).eval().to(args.device)
    if model.config.hidden_size != 1280:
        parser.error("Expected the 1,280-dimensional ESM2-650M model")
    vectors = np.empty((len(inventory), 1280), dtype=np.float32)
    with torch.no_grad():
        for index, sequence in enumerate(inventory.Sequence):
            tokens = tokenizer(sequence, return_tensors="pt").to(args.device)
            eos = (tokens.input_ids[0] == tokenizer.eos_token_id).nonzero().item()
            hidden = model(**tokens, output_hidden_states=True).hidden_states[-1]
            vectors[index] = hidden[0, 1:eos].mean(dim=0).cpu().numpy()
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite sequence embedding values")
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.save_path, ids=inventory.Entry.to_numpy(dtype=str), embeddings=vectors)


if __name__ == "__main__":
    main()
