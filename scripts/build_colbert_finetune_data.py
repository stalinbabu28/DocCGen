from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "create", "do", "does", "for", "from", "get", "give", "if", "in", "into",
    "is", "it", "let", "make", "me", "need", "of", "on", "or", "run", "set",
    "show", "that", "the", "this", "to", "try", "use", "using", "with", "you",
    "your", "ensure", "present", "running", "condition", "file", "job",
}

INFO_CUES = {
    "info", "information", "show", "list", "details", "describe", "display", "get"
}


def normpath(p: str) -> str:
    return os.path.normpath(os.path.abspath(p))


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9_./-]+", (text or "").lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 1]


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_doc_map(doc_map_path: Path):
    with doc_map_path.open("r", encoding="utf-8") as f:
        doc_map = json.load(f)

    if not isinstance(doc_map, list):
        raise ValueError(f"Expected doc_map to be a list, got {type(doc_map)}")

    by_pid: Dict[int, dict] = {}
    by_source: Dict[str, dict] = {}
    by_module: Dict[str, dict] = {}
    by_collection: Dict[str, List[dict]] = defaultdict(list)

    for item in doc_map:
        pid = int(item["pid"])
        module = str(item["module"])
        collection = ".".join(module.split(".")[:-1])

        by_pid[pid] = item
        by_source[normpath(item["source"])] = item
        by_module[module] = item
        by_collection[collection].append(item)

    return doc_map, by_pid, by_source, by_module, by_collection


def load_old_benchmark_negatives(old_output_dir: Path) -> Dict[str, List[str]]:
    """
    Reads the old benchmark output folder and collects wrong generated modules
    per query. This gives us real hard negatives without calling ColBERT.
    """
    query_to_neg_modules: Dict[str, List[str]] = defaultdict(list)

    if not old_output_dir.exists():
        return query_to_neg_modules

    for path in sorted(old_output_dir.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        if not isinstance(obj, dict):
            continue

        query = (obj.get("query") or "").strip()
        expected = (obj.get("expected_module") or "").strip()
        generated = (obj.get("generated_module") or "").strip()

        if not query or not generated:
            continue

        if expected and generated == expected:
            continue

        if generated not in query_to_neg_modules[query]:
            query_to_neg_modules[query].append(generated)

    return query_to_neg_modules


def find_positive(sample: dict, by_source: Dict[str, dict], by_module: Dict[str, dict]) -> Optional[dict]:
    doc_path = sample.get("document")
    if doc_path:
        pos = by_source.get(normpath(doc_path))
        if pos is not None:
            return pos

    expected_module = sample.get("expected_module")
    if expected_module:
        return by_module.get(expected_module)

    return None


def query_has_info_cue(query_tokens: List[str]) -> bool:
    return any(tok in INFO_CUES for tok in query_tokens)


def doc_text_for_scoring(doc: dict) -> str:
    parts = [
        str(doc.get("module", "")),
        str(doc.get("short_description", "")),
        str(doc.get("text", "")),
    ]
    return " ".join(parts)


def build_indexes(by_pid: Dict[int, dict]):
    """
    Precompute token sets once, so the script stays fast.
    """
    doc_tokens: Dict[int, set[str]] = {}
    token_to_pids: Dict[str, set[int]] = defaultdict(set)

    for pid, doc in by_pid.items():
        toks = set(tokenize(doc_text_for_scoring(doc)))
        doc_tokens[pid] = toks
        for tok in toks:
            token_to_pids[tok].add(pid)

    return doc_tokens, token_to_pids


def score_doc(
    query_tokens: List[str],
    doc: dict,
    doc_tokens: Dict[int, set[str]],
    positive_collection: Optional[str],
) -> float:
    pid = int(doc["pid"])
    toks = doc_tokens.get(pid, set())

    if not toks:
        return 0.0

    qset = set(query_tokens)
    overlap = len(qset & toks)

    module = str(doc.get("module", "")).lower()
    short_desc = str(doc.get("short_description", "")).lower()
    score = float(overlap)

    for qt in query_tokens:
        if qt in module:
            score += 1.5
        if qt in short_desc:
            score += 0.75

    if positive_collection:
        doc_collection = ".".join(str(doc.get("module", "")).split(".")[:-1])
        if doc_collection == positive_collection and str(doc.get("module", "")) != positive_collection:
            score += 0.75

    if query_has_info_cue(query_tokens) and module.endswith("_info"):
        score += 0.35

    return score


def mine_fallback_negatives(
    query: str,
    positive: dict,
    by_pid: Dict[int, dict],
    by_collection: Dict[str, List[dict]],
    doc_tokens: Dict[int, set[str]],
    token_to_pids: Dict[str, set[int]],
    num_negs: int,
) -> List[dict]:
    """
    Fast lexical miner:
    - query-token overlap
    - same-collection docs
    - no full-corpus brute force
    """
    query_tokens = tokenize(query)
    positive_collection = ".".join(str(positive["module"]).split(".")[:-1])
    qset = set(query_tokens)

    candidate_pids: set[int] = set()
    for tok in qset:
        candidate_pids.update(token_to_pids.get(tok, set()))

    for doc in by_collection.get(positive_collection, []):
        candidate_pids.add(int(doc["pid"]))

    if len(candidate_pids) < max(20, num_negs * 4):
        for pid in list(by_pid.keys())[:500]:
            candidate_pids.add(pid)

    scored: List[Tuple[float, int, dict]] = []
    for pid in candidate_pids:
        doc = by_pid.get(pid)
        if doc is None:
            continue
        if int(doc["pid"]) == int(positive["pid"]):
            continue
        if doc["module"] == positive["module"]:
            continue

        score = score_doc(query_tokens, doc, doc_tokens, positive_collection)
        scored.append((score, pid, doc))

    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    negs: List[dict] = []
    seen_modules = set()
    for _, _, doc in scored:
        if doc["module"] in seen_modules:
            continue
        negs.append(doc)
        seen_modules.add(doc["module"])
        if len(negs) >= num_negs:
            break

    return negs


def resolve_negatives_for_query(
    query: str,
    positive: dict,
    by_pid: Dict[int, dict],
    by_module: Dict[str, dict],
    by_collection: Dict[str, List[dict]],
    doc_tokens: Dict[int, set[str]],
    token_to_pids: Dict[str, set[int]],
    old_query_to_neg_modules: Dict[str, List[str]],
    num_negs: int,
) -> List[dict]:
    negs: List[dict] = []
    seen_modules = {positive["module"]}
    seen_pids = {int(positive["pid"])}

    # 1) Real mistakes from your old benchmark output
    for neg_module in old_query_to_neg_modules.get(query, []):
        doc = by_module.get(neg_module)
        if doc is None:
            continue
        pid = int(doc["pid"])
        if pid in seen_pids:
            continue
        if doc["module"] in seen_modules:
            continue
        negs.append(doc)
        seen_modules.add(doc["module"])
        seen_pids.add(pid)
        if len(negs) >= num_negs:
            return negs

    # 2) Fast lexical / same-collection hard negatives
    fallback = mine_fallback_negatives(
        query=query,
        positive=positive,
        by_pid=by_pid,
        by_collection=by_collection,
        doc_tokens=doc_tokens,
        token_to_pids=token_to_pids,
        num_negs=num_negs,
    )

    for doc in fallback:
        pid = int(doc["pid"])
        if pid in seen_pids:
            continue
        if doc["module"] in seen_modules:
            continue
        negs.append(doc)
        seen_modules.add(doc["module"])
        seen_pids.add(pid)
        if len(negs) >= num_negs:
            break

    return negs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, help="Benchmark JSONL (typically train.jsonl)")
    parser.add_argument("--old-output-dir", default=None)
    parser.add_argument("--doc-map", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-negs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    benchmark_path = Path(args.benchmark)
    doc_map_path = Path(args.doc_map)
    out_dir = Path(args.out_dir)
    old_output_dir = (
        Path(args.old_output_dir).resolve()
        if args.old_output_dir
        else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_jsonl(benchmark_path)
    rng.shuffle(samples)

    print()
    print("=" * 70)
    print("Building ColBERT training data")
    print("=" * 70)
    print(f"Input benchmark : {benchmark_path}")
    print(f"Queries         : {len(samples)}")
    print(f"Negatives/query : {args.num_negs}")
    print()

    _, by_pid, by_source, by_module, by_collection = load_doc_map(doc_map_path)
    old_query_to_neg_modules = (
        load_old_benchmark_negatives(old_output_dir)
        if old_output_dir
        else {}
    )
    doc_tokens, token_to_pids = build_indexes(by_pid)

    collection_path = out_dir / "collection.tsv"
    queries_path = out_dir / "queries.train.tsv"
    triples_path = out_dir / "triples.train.tsv"
    debug_path = out_dir / "negatives.debug.jsonl"

    # IMPORTANT:
    # Collection pids must be 0..N-1 in exact line order for this ColBERT fork.
    # Keep a mapping from original pid -> new sequential pid.
    old_to_new_pid: Dict[int, int] = {}
    new_to_old_pid: Dict[int, int] = {}

    with collection_path.open("w", encoding="utf-8") as f:
        for new_pid, old_pid in enumerate(sorted(by_pid)):
            old_to_new_pid[old_pid] = new_pid
            new_to_old_pid[new_pid] = old_pid

            item = by_pid[old_pid]
            text = str(item["text"]).replace("\t", " ").replace("\n", " ").strip()
            f.write(f"{new_pid}\t{text}\n")

    qid = 0
    query_count = 0
    triple_count = 0

    with queries_path.open("w", encoding="utf-8") as qf, \
         triples_path.open("w", encoding="utf-8") as tf, \
         debug_path.open("w", encoding="utf-8") as df:

        for sample in samples:
            query = (sample.get("query") or "").strip()
            if not query:
                continue

            positive = find_positive(sample, by_source, by_module)
            if positive is None:
                continue

            old_pos_pid = int(positive["pid"])
            if old_pos_pid not in old_to_new_pid:
                continue

            qid += 1
            query_count += 1
            qf.write(f"{qid}\t{query}\n")

            negatives = resolve_negatives_for_query(
                query=query,
                positive=positive,
                by_pid=by_pid,
                by_module=by_module,
                by_collection=by_collection,
                doc_tokens=doc_tokens,
                token_to_pids=token_to_pids,
                old_query_to_neg_modules=old_query_to_neg_modules,
                num_negs=args.num_negs,
            )

            if len(negatives) < args.num_negs:
                pool = [
                    x for x in by_pid.values()
                    if int(x["pid"]) != old_pos_pid
                    and x["module"] != positive["module"]
                ]
                rng.shuffle(pool)
                neg_ids = {int(n["pid"]) for n in negatives}
                for x in pool:
                    if len(negatives) >= args.num_negs:
                        break
                    if int(x["pid"]) not in neg_ids:
                        negatives.append(x)
                        neg_ids.add(int(x["pid"]))

            negatives = negatives[:args.num_negs]

            # Write one JSON example per query:
            # [qid, positive_pid, neg1, neg2, ..., negN]
            example = [qid, old_to_new_pid[old_pos_pid]]
            remapped_negs = []

            for neg in negatives:
                old_neg_pid = int(neg["pid"])
                if old_neg_pid not in old_to_new_pid:
                    continue
                new_neg_pid = old_to_new_pid[old_neg_pid]
                if new_neg_pid == old_to_new_pid[old_pos_pid]:
                    continue
                example.append(new_neg_pid)
                remapped_negs.append(new_neg_pid)

            # if some negatives were skipped by remap, keep filling from the pool
            if len(example) < 2 + args.num_negs:
                needed = (2 + args.num_negs) - len(example)
                extra_pool = [
                    x for x in by_pid.values()
                    if int(x["pid"]) != old_pos_pid
                    and x["module"] != positive["module"]
                    and old_to_new_pid.get(int(x["pid"])) is not None
                    and old_to_new_pid[int(x["pid"])] not in example
                ]
                rng.shuffle(extra_pool)
                for x in extra_pool:
                    if needed <= 0:
                        break
                    example.append(old_to_new_pid[int(x["pid"])])
                    remapped_negs.append(old_to_new_pid[int(x["pid"])])
                    needed -= 1

            # Final safety trim
            example = example[: 2 + args.num_negs]

            tf.write(json.dumps(example))
            tf.write("\n")
            triple_count += 1

            df.write(
                json.dumps(
                    {
                        "qid": qid,
                        "query": query,
                        "old_positive_pid": old_pos_pid,
                        "new_positive_pid": old_to_new_pid[old_pos_pid],
                        "positive_module": positive["module"],
                        "negative_old_pids": [int(n["pid"]) for n in negatives],
                        "negative_new_pids": remapped_negs,
                        "negative_modules": [n["module"] for n in negatives],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {query_count} queries to {queries_path}")
    print(f"Wrote {triple_count} training examples to {triples_path}")
    print(f"Wrote collection to {collection_path}")
    print(f"Wrote debug negatives to {debug_path}")
    if old_output_dir:
        print(f"Used old benchmark outputs from: {old_output_dir}")
    else:
        print("No old benchmark outputs supplied.")


if __name__ == "__main__":
    main()