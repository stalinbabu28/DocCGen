#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from colbert import Searcher
from colbert.infra import Run, RunConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INDEX_ROOT = PROJECT_ROOT / "experiments"
EXPERIMENT = "phase3_colbert"
INDEX_NAME = "ansible_docs"

DOC_MAP = PROJECT_ROOT / "colbert_data" / "data" / "doc_map.json"


def load_doc_map():
    with open(DOC_MAP, "r") as f:
        return json.load(f)


def print_result(rank, pid, score, doc):
    print("=" * 100)
    print(f"Rank   : {rank}")
    print(f"PID    : {pid}")
    print(f"Score  : {score:.4f}")
    print(f"Module : {doc['module']}")
    print(f"Chunk  : {doc['chunk']}")
    print(f"Source : {doc['source']}")
    print("-" * 100)

    text = doc["text"]

    if len(text) > 1200:
        text = text[:1200] + "\n..."

    print(text)
    print()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    doc_map = load_doc_map()

    print("\nLoading ColBERT index...\n")

    with Run().context(
        RunConfig(
            root=str(INDEX_ROOT),
            experiment=EXPERIMENT,
        )
    ):

        searcher = Searcher(index=INDEX_NAME)

        print("Searching...\n")

        results = searcher.search(
            args.query,
            k=args.topk,
        )

    print("Raw result type:", type(results))
    print()

    #
    # Handle different ColBERT versions
    #

    if len(results) == 3:

        pids, ranks, scores = results

    elif len(results) == 2:

        pids, scores = results
        ranks = list(range(1, len(pids) + 1))

    else:
        raise RuntimeError(f"Unknown ColBERT output: {results}")

    print("=" * 100)
    print("QUERY")
    print("=" * 100)
    print(args.query)
    print()

    for rank, pid, score in zip(ranks, pids, scores):

        #
        # doc_map uses pid starting at 1
        #

        if isinstance(pid, str):
            pid = int(pid)

        doc = doc_map[pid - 1]

        print_result(rank, pid, score, doc)


if __name__ == "__main__":
    main()