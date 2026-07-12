#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--collection",
        required=True,
    )

    parser.add_argument(
        "--index-root",
        required=True,
    )

    parser.add_argument(
        "--experiment",
        default="phase3_colbert",
        help="Experiment folder (phase3_colbert / phase3_colbert_ft)",
    )

    parser.add_argument(
        "--name",
        required=True,
    )

    parser.add_argument(
        "--doc-maxlen",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--nbits",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--nranks",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(args.index_root).resolve()

    index_path = (
        root
        / args.experiment
        / "indexes"
        / args.name
    )

    if args.overwrite and index_path.exists():
        shutil.rmtree(index_path)

    with Run().context(

        RunConfig(
            root=str(root),
            experiment=args.experiment,
            nranks=args.nranks,
        )

    ):

        config = ColBERTConfig(

            root=str(root),

            experiment=args.experiment,

            checkpoint=args.checkpoint,

            doc_maxlen=args.doc_maxlen,

            dim=args.dim,

            nbits=args.nbits,

            nranks=args.nranks,

            gpus=1,

            amp=True,

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

    print()

    print("=" * 70)

    print("Finished indexing")

    print("=" * 70)

    print("Experiment :", args.experiment)

    print("Index name :", args.name)

    print("Location   :", index_path)

    print()


if __name__ == "__main__":
    main()