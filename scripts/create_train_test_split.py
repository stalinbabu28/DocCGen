from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")


def fingerprint(rows):

    m = hashlib.sha256()

    for r in rows:
        m.update(
            json.dumps(
                r,
                sort_keys=True,
            ).encode()
        )

    return m.hexdigest()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        required=True,
    )

    parser.add_argument(
        "--out-dir",
        required=True,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rows = read_jsonl(args.benchmark)

    rng = random.Random(args.seed)

    rng.shuffle(rows)

    n = len(rows)

    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)

    train = rows[:n_train]
    valid = rows[n_train:n_train + n_valid]
    test = rows[n_train + n_valid:]

    out = Path(args.out_dir)

    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "valid.jsonl", valid)
    write_jsonl(out / "test.jsonl", test)

    manifest = {
        "seed": args.seed,
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "benchmark_sha256": fingerprint(rows),
    }

    with open(
        out / "split_manifest.json",
        "w",
        encoding="utf8",
    ) as f:
        json.dump(manifest, f, indent=2)

    print()

    print("Split complete")

    print("----------------")

    print("Train :", len(train))
    print("Valid :", len(valid))
    print("Test  :", len(test))

    print()

    print("Manifest written to")

    print(out / "split_manifest.json")


if __name__ == "__main__":
    main()