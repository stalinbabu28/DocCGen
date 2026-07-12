from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--index-root", required=True)

    parser.add_argument("--experiment", default="phase3_colbert")
    parser.add_argument("--name", default="phase3_colbert_ft")

    parser.add_argument("--doc-maxlen", type=int, default=256)
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    index_root = Path(args.index_root).resolve()

    index_path = (
        index_root
        / args.experiment
        / "indexes"
        / args.name
    )

    if args.overwrite and index_path.exists():
        shutil.rmtree(index_path)

    with Run().context(
        RunConfig(
            root=str(index_root),
            experiment=args.experiment,
            nranks=1,
        )
    ):
        config = ColBERTConfig(
            root=str(index_root),
            experiment=args.experiment,
            doc_maxlen=args.doc_maxlen,
            nbits=args.nbits,
        )

        indexer = Indexer(
            checkpoint=args.checkpoint,
            config=config,
        )

        indexer.index(
            name=args.name,
            collection=args.collection,
            overwrite=args.overwrite,
        )

    print("\n" + "=" * 70)
    print("Finished indexing")
    print("=" * 70)
    print(f"Experiment : {args.experiment}")
    print(f"Index name : {args.name}")
    print(f"Location   : {index_path}")


if __name__ == "__main__":
    main()