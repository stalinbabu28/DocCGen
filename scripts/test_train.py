from __future__ import annotations

import os
import torch

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.training.training import train
from colbert.utils.distributed import init as colbert_init


def main() -> None:
    # Single-process DDP setup
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    nranks, _ = colbert_init(rank=0)

    config = ColBERTConfig(
        checkpoint="colbert_data/checkpoints/colbertv2.0",
        root="/home/dao-lab/stalin/phase3withcolbert/experiments",
        experiment="default",
        bsize=16,
        accumsteps=1,
        lr=3e-6,
        maxsteps=1,
        warmup=None,
        doc_maxlen=220,
        nway=2,
        dim=128,
        query_maxlen=32,
        mask_punctuation=True,
        similarity="cosine",
        use_ib_negatives=False,
        amp=True,
        nranks=nranks,
        rank=0,
        gpus=max(1, torch.cuda.device_count()),
    )

    with Run().context(
        RunConfig(
            root="/home/dao-lab/stalin/phase3withcolbert/experiments",
            experiment="default",
            nranks=nranks,
        )
    ):
        train(
            config=config,
            triples="/home/dao-lab/stalin/phase3withcolbert/colbert_ft/data/triples.train.tsv",
            queries="/home/dao-lab/stalin/phase3withcolbert/colbert_ft/data/queries.train.tsv",
            collection="/home/dao-lab/stalin/phase3withcolbert/colbert_ft/data/collection.tsv",
        )


if __name__ == "__main__":
    main()