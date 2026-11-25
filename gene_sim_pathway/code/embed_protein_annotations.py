"""Embed UniProt GO annotations with DashScope text-embedding-v4."""
import argparse
import gzip
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open_path", type=Path, required=True, help="Accession-to-GO JSON or source XLSX")
    parser.add_argument("--save_path", type=Path, required=True)
    parser.add_argument("--prompts_only", action="store_true", help="Export input texts without requesting embeddings")
    args = parser.parse_args()
    if args.open_path.suffix.lower() == ".xlsx":
        import pandas as pd

        table = pd.read_excel(args.open_path, dtype=str)
        annotations = table.GO_annotation.fillna("Unknow")
        annotations = annotations.where(annotations.str.strip().ne(""), "Unknow")
        data = dict(zip(table.protein_ID, annotations))
    elif args.open_path.suffix.lower() == ".gz":
        with gzip.open(args.open_path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
    else:
        data = json.loads(args.open_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data or any(
        not isinstance(uid, str) or not uid.strip() or (value is not None and not isinstance(value, str))
        for uid, value in data.items()
    ):
        parser.error("Expected a nonempty accession-to-annotation mapping")
    prompts = {uid: f"Uniprot ID: {uid}; {annotation or 'Unknown'}" for uid, annotation in data.items()}
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    if args.prompts_only:
        args.save_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    import numpy as np
    from openai import OpenAI

    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        parser.error("Set DASHSCOPE_API_KEY before requesting embeddings")
    client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    contents = list(prompts.values())
    vectors = np.empty((len(contents), 1024), dtype=np.float32)
    for start in range(0, len(contents), 10):
        batch = contents[start:start + 10]
        response = client.embeddings.create(
            model="text-embedding-v4", input=batch, dimensions=1024, encoding_format="float"
        )
        if sorted(row.index for row in response.data) != list(range(len(batch))):
            raise ValueError("Embedding response does not cover every input")
        for row in response.data:
            vectors[start + row.index] = row.embedding
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite embedding values")
    np.savez_compressed(args.save_path, ids=np.asarray(list(prompts)), embeddings=vectors)


if __name__ == "__main__":
    main()
