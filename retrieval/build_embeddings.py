from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer


def _iter_docs(docs_dir: Path):
    for path in docs_dir.rglob("*.json"):
        yield path


def _build_text(doc: dict) -> str:
    module = doc.get("module", "")
    collection = doc.get("collection", "")
    short_desc = doc.get("short_description", "")
    description = " ".join(doc.get("description", []) or [])
    options = doc.get("options", {}) or {}

    required = []
    all_options = []
    for name, meta in options.items():
        all_options.append(name)
        if isinstance(meta, dict) and meta.get("required", False):
            required.append(name)

    module_keywords = " ".join(module.replace("_", " ").split())
    collection_text = collection.replace(".", " ")

    return f"""
Collection: {collection_text}
Module: {module}
Module Keywords: {module_keywords}
Short Description: {short_desc}
Description: {description}
Required Parameters: {' '.join(required)}
All Parameters: {' '.join(all_options)}
""".strip()


def main():
    parser = argparse.ArgumentParser(description="Build Ansible embeddings for retrieval.")
    parser.add_argument("--docs-dir", default=os.getenv("ANSIBLE_DOCS_DIR", "/home/dao-lab/stalin/docs"))
    parser.add_argument("--output", default=os.getenv("ANSIBLE_EMBEDDINGS_FILE", "ansible_embeddings.pkl"))
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    model = SentenceTransformer(args.model)
    database = []

    files = list(_iter_docs(docs_dir))
    print(f"Processing {len(files)} docs...")

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue

        text = _build_text(doc)
        emb = model.encode(text)

        database.append({
            "path": str(path.resolve()),
            "collection": doc.get("collection", ""),
            "module": doc.get("module", ""),
            "text": text,
            "embedding": emb,
        })

    with out_path.open("wb") as f:
        pickle.dump(database, f)

    print(f"Saved {len(database)} embeddings to {out_path}")


if __name__ == "__main__":
    main()