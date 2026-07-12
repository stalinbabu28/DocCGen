from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.training.training import train
from colbert.utils.distributed import init as colbert_init


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="ColBERT experiment root")
    parser.add_argument("--experiment", default="phase3_colbert_ft")
    parser.add_argument("--checkpoint", required=True, help="Base checkpoint path")
    parser.add_argument("--triples", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--collection", required=True)

    parser.add_argument("--bsize", type=int, default=16)
    parser.add_argument("--accumsteps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--maxsteps", type=int, default=5000)
    parser.add_argument("--doc-maxlen", type=int, default=256)
    parser.add_argument("--nway", type=int, default=9)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    # Single-process DDP setup so ColBERT's training code can wrap the model.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    nranks, _ = colbert_init(rank=0)

    root = Path(args.root).resolve()

    with Run().context(
        RunConfig(
            root=str(root),
            experiment=args.experiment,
            nranks=nranks,
        )
    ):
        config = ColBERTConfig(
            root=str(root),
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            bsize=args.bsize,
            accumsteps=args.accumsteps,
            lr=args.lr,
            warmup=args.warmup,
            maxsteps=args.maxsteps,
            doc_maxlen=args.doc_maxlen,
            nway=args.nway,
            dim=args.dim,
            query_maxlen=32,
            mask_punctuation=True,
            similarity="cosine",
            use_ib_negatives=True,
            amp=args.amp,
            nranks=nranks,
            rank=0,
            gpus=max(1, torch.cuda.device_count()),
        )

        out = train(
            config=config,
            triples=args.triples,
            queries=args.queries,
            collection=args.collection,
        )

    print("\nTraining finished.")
    if out is not None:
        print("Returned:", out)


if __name__ == "__main__":
    main()