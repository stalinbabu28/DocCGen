#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from colbert import Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig


def load_doc_map(path: Path) -> Dict[int, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, Any]] = {}
    for row in data:
        try:
            out[int(row["pid"])] = row
        except Exception:
            continue
    return out


def extract_rows(ranking: Any) -> Optional[List[Tuple[int, float]]]:
    # Common ColBERT ranking shapes across versions
    if ranking is None:
        return None

    if isinstance(ranking, list):
        rows: List[Tuple[int, float]] = []
        for item in ranking:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                try:
                    rows.append((int(item[0]), float(item[1])))
                except Exception:
                    pass
            elif isinstance(item, dict):
                pid = item.get("pid", item.get("docid", item.get("id")))
                score = item.get("score", item.get("scores"))
                if pid is not None:
                    try:
                        rows.append((int(pid), float(score) if score is not None else 0.0))
                    except Exception:
                        pass
        return rows or None

    if hasattr(ranking, "pids"):
        pids = list(getattr(ranking, "pids"))
        scores = list(getattr(ranking, "scores", [0.0] * len(pids)))
        rows = []
        for pid, score in zip(pids, scores):
            try:
                rows.append((int(pid), float(score)))
            except Exception:
                continue
        return rows or None

    if hasattr(ranking, "data"):
        data = getattr(ranking, "data")
        if isinstance(data, list):
            rows = []
            for item in data:
                if isinstance(item, dict):
                    pid = item.get("pid", item.get("docid", item.get("id")))
                    score = item.get("score", 0.0)
                    if pid is not None:
                        try:
                            rows.append((int(pid), float(score)))
                        except Exception:
                            pass
            return rows or None

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--doc-map", required=True)
    ap.add_argument("--experiment", default="phase3_colbert")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--query", action="append", help="Repeatable. If omitted, uses a few smoke queries.")
    args = ap.parse_args()

    doc_map = load_doc_map(Path(args.doc_map))
    queries = args.query or [
        "Copy a file to demo-dest",
        "Download demo-url to demo-dest",
        "Create a file at demo-path",
    ]

    with Run().context(RunConfig(nranks=1, experiment=args.experiment)):
        config = ColBERTConfig(root=str(Path(args.index_root)))
        searcher = Searcher(index=args.name, config=config)

        for q in queries:
            print("=" * 80)
            print("QUERY:", q)
            try:
                ranking = searcher.search(q, k=args.k)
            except Exception as e:
                print("SEARCH FAILED:", repr(e))
                continue

            rows = extract_rows(ranking)
            if not rows:
                print("RAW RESULT:", ranking)
                continue

            for rank, (pid, score) in enumerate(rows[: args.k], start=1):
                meta = doc_map.get(pid, {})
                module = meta.get("module", "<unknown>")
                source = meta.get("source", "<unknown>")
                print(f"{rank:>2}. pid={pid:<6} score={score:<10.4f} module={module} source={source}")


if __name__ == "__main__":
    main()