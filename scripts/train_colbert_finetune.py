from __future__ import annotations

import argparse
from pathlib import Path

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.training.training import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--experiment", default="phase3_colbert_ft")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--triples", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--collection", required=True)

    parser.add_argument("--bsize", type=int, default=16)
    parser.add_argument("--accumsteps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--maxsteps", type=int, default=5000)
    parser.add_argument("--doc-maxlen", type=int, default=256)
    parser.add_argument("--nway", type=int, default=64)
    parser.add_argument("--dim", type=int, default=128)

    args = parser.parse_args()

    root = Path(args.root).resolve()

    with Run().context(
        RunConfig(
            root=str(root),
            experiment=args.experiment,
            nranks=1,
        )
    ):
        config = ColBERTConfig(
            checkpoint=args.checkpoint,
            root=str(root),
            experiment=args.experiment,
            doc_maxlen=args.doc_maxlen,
            bsize=args.bsize,
            accumsteps=args.accumsteps,
            lr=args.lr,
            warmup=args.warmup,
            maxsteps=args.maxsteps,
            nway=args.nway,
            dim=args.dim,
            similarity="cosine",
            use_ib_negatives=True,
            query_maxlen=32,
            mask_punctuation=True,
            amp=True,
            nranks=1,
            rank=0,
            gpus=1,
        )

        ckpt_path = train(
            config,
            triples=args.triples,
            queries=args.queries,
            collection=args.collection,
        )

    print(f"Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()