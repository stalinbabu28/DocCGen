from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from evaluation.retrieval_metrics import (
    canonical_id,
    hits_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


METRIC_KS = (1, 5, 10)
RETRIEVAL_K = 20


@dataclass
class QueryResult:
    query: str
    gold_module: str
    retrieved: List[str]
    gold_rank: Optional[int]
    rr: float
    hits: Dict[int, float]
    recalls: Dict[int, float]
    precisions: Dict[int, float]
    ndcgs: Dict[int, float]
    retrieval_ms: float


@dataclass
class RetrieverSummary:
    retriever: str
    total_samples: int
    top1: float
    top5: float
    top10: float
    recall1: float
    recall5: float
    recall10: float
    precision1: float
    precision5: float
    precision10: float
    mrr: float
    ndcg1: float
    ndcg5: float
    ndcg10: float
    avg_retrieval_ms: float


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_doc_map(doc_map_path: Path) -> List[dict]:
    with doc_map_path.open("r", encoding="utf-8") as f:
        doc_map = json.load(f)

    if not isinstance(doc_map, list):
        raise ValueError(f"doc_map must be a list, got {type(doc_map)}")

    return doc_map


def build_indexed_docs(doc_map: List[dict]) -> List[dict]:
    """
    The ColBERT collection was written in ascending old-pid order.
    This function recreates the same 0-based index order so that
    returned ColBERT pids can be mapped back to modules correctly.
    """
    return [doc for doc in sorted(doc_map, key=lambda d: int(d["pid"]))]


def normalize_gold(sample: dict) -> str:
    gold = sample.get("expected_module") or sample.get("module") or ""
    return canonical_id(gold)


def gold_relevance_map(gold_module: str) -> Dict[str, float]:
    return {canonical_id(gold_module): 1.0}


def candidate_module_id(candidate: dict) -> str:
    module = candidate.get("module")
    if isinstance(module, str) and module.strip():
        return canonical_id(module)

    module_fqn = candidate.get("module_fqn")
    if isinstance(module_fqn, str) and module_fqn.strip():
        return canonical_id(module_fqn.split(".")[-1])

    path = candidate.get("path")
    if isinstance(path, str) and path.strip():
        return canonical_id(Path(path).stem)

    return ""


def build_hybrid_backend():
    """
    Uses your existing hybrid retrieval stack.
    We lazy-import so this script still loads even if the hybrid stack
    is temporarily broken.
    """
    from pipeline.select_module_candidates import get_ranked_candidates

    def retrieve(query: str, k: int) -> List[dict]:
        return get_ranked_candidates(query, k=k)

    return retrieve


def build_colbert_backend(
    project_root: Path,
    experiment: str,
    index_name: str,
):
    """
    Uses the current ColBERT Searcher directly.
    """
    from colbert import Searcher
    from colbert.infra import Run, RunConfig

    run_ctx = Run().context(
        RunConfig(
            root=str(project_root),
            experiment=experiment,
            nranks=1,
        )
    )

    # Keep the context object alive inside the closure.
    with run_ctx:
        searcher = Searcher(index=index_name)

    def retrieve(query: str, k: int) -> List[dict]:
        results = searcher.search(query, k=k)

        if isinstance(results, tuple):
            if len(results) == 3:
                pids, ranks, scores = results
            elif len(results) == 2:
                pids, scores = results
                ranks = list(range(1, len(pids) + 1))
            else:
                raise RuntimeError(f"Unexpected ColBERT search result tuple length: {len(results)}")
        else:
            raise RuntimeError(f"Unexpected ColBERT search result type: {type(results)}")

        out: List[dict] = []
        for pid, rank, score in zip(pids, ranks, scores):
            out.append(
                {
                    "pid": int(pid),
                    "rank": int(rank),
                    "score": float(score),
                }
            )
        return out

    return retrieve


def retrieve_module_ids_from_hybrid(query: str, k: int) -> List[str]:
    candidates = build_hybrid_backend()(query, k)
    return [candidate_module_id(c) for c in candidates if candidate_module_id(c)]


def retrieve_module_ids_from_colbert(
    query: str,
    k: int,
    backend_retrieve: Callable[[str, int], List[dict]],
    indexed_docs: List[dict],
) -> List[str]:
    results = backend_retrieve(query, k)
    module_ids: List[str] = []

    for item in results:
        pid = int(item["pid"])
        if pid < 0 or pid >= len(indexed_docs):
            continue
        module_ids.append(canonical_id(indexed_docs[pid]["module"]))

    return module_ids


def evaluate_query(
    query: str,
    gold_module: str,
    retrieve_fn: Callable[[str, int], List[str]],
    k: int = RETRIEVAL_K,
) -> QueryResult:
    start = time.perf_counter()
    retrieved_ids = retrieve_fn(query, k)
    retrieval_ms = (time.perf_counter() - start) * 1000.0

    gold = canonical_id(gold_module)
    gold_set = [gold]

    gold_rank: Optional[int] = None
    for idx, mid in enumerate(retrieved_ids, start=1):
        if canonical_id(mid) == gold:
            gold_rank = idx
            break

    hits: Dict[int, float] = {}
    recalls: Dict[int, float] = {}
    precisions: Dict[int, float] = {}
    ndcgs: Dict[int, float] = {}

    for kk in METRIC_KS:
        hits[kk] = hits_at_k(retrieved_ids, gold_set, kk)
        recalls[kk] = recall_at_k(retrieved_ids, gold_set, kk)
        precisions[kk] = precision_at_k(retrieved_ids, gold_set, kk)
        ndcgs[kk] = ndcg_at_k(retrieved_ids, gold_relevance_map(gold_module), kk)

    rr = reciprocal_rank(retrieved_ids, gold_set, k=k)

    return QueryResult(
        query=query,
        gold_module=gold,
        retrieved=retrieved_ids,
        gold_rank=gold_rank,
        rr=rr,
        hits=hits,
        recalls=recalls,
        precisions=precisions,
        ndcgs=ndcgs,
        retrieval_ms=retrieval_ms,
    )


def summarize(retriever_name: str, results: Sequence[QueryResult]) -> RetrieverSummary:
    n = len(results) if results else 1

    return RetrieverSummary(
        retriever=retriever_name,
        total_samples=len(results),
        top1=sum(r.hits[1] for r in results) / n,
        top5=sum(r.hits[5] for r in results) / n,
        top10=sum(r.hits[10] for r in results) / n,
        recall1=sum(r.recalls[1] for r in results) / n,
        recall5=sum(r.recalls[5] for r in results) / n,
        recall10=sum(r.recalls[10] for r in results) / n,
        precision1=sum(r.precisions[1] for r in results) / n,
        precision5=sum(r.precisions[5] for r in results) / n,
        precision10=sum(r.precisions[10] for r in results) / n,
        mrr=mean_reciprocal_rank([r.rr for r in results]) if results else 0.0,
        ndcg1=sum(r.ndcgs[1] for r in results) / n,
        ndcg5=sum(r.ndcgs[5] for r in results) / n,
        ndcg10=sum(r.ndcgs[10] for r in results) / n,
        avg_retrieval_ms=mean(r.retrieval_ms for r in results) if results else 0.0,
    )


def print_summary(summary: RetrieverSummary) -> None:
    print("\n" + "=" * 100)
    print(f"Retriever: {summary.retriever}")
    print("=" * 100)
    print(f"Total samples   : {summary.total_samples}")
    print(f"Top1            : {summary.top1:.4f}")
    print(f"Top5            : {summary.top5:.4f}")
    print(f"Top10           : {summary.top10:.4f}")
    print(f"Recall@1        : {summary.recall1:.4f}")
    print(f"Recall@5        : {summary.recall5:.4f}")
    print(f"Recall@10       : {summary.recall10:.4f}")
    print(f"Precision@1     : {summary.precision1:.4f}")
    print(f"Precision@5     : {summary.precision5:.4f}")
    print(f"Precision@10    : {summary.precision10:.4f}")
    print(f"MRR             : {summary.mrr:.4f}")
    print(f"nDCG@1          : {summary.ndcg1:.4f}")
    print(f"nDCG@5          : {summary.ndcg5:.4f}")
    print(f"nDCG@10         : {summary.ndcg10:.4f}")
    print(f"Avg retrieval ms : {summary.avg_retrieval_ms:.2f}")


def write_outputs(
    out_dir: Path,
    retriever_name: str,
    results: Sequence[QueryResult],
    summary: RetrieverSummary,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = out_dir / f"{retriever_name}_results.jsonl"
    summary_json = out_dir / f"{retriever_name}_summary.json"
    summary_csv = out_dir / f"{retriever_name}_summary.csv"

    with results_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summary).keys()))
        writer.writeheader()
        writer.writerow(asdict(summary))


def evaluate_retriever(
    retriever_name: str,
    samples: Sequence[dict],
    retrieve_fn: Callable[[str, int], List[str]],
) -> Tuple[List[QueryResult], RetrieverSummary]:
    results: List[QueryResult] = []

    for i, sample in enumerate(samples, start=1):
        query = (sample.get("query") or "").strip()
        gold_module = sample.get("expected_module") or ""

        if not query or not gold_module:
            continue

        result = evaluate_query(query, gold_module, retrieve_fn)
        results.append(result)

        print(f"[{retriever_name}] {i}/{len(samples)}  {query}")

    summary = summarize(retriever_name, results)
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        help="Path to split JSONL, usually colbert_ft/splits/test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write per-retriever results",
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["hybrid", "colbert_zero", "colbert_ft"],
        choices=["hybrid", "colbert_zero", "colbert_ft"],
        help="Which retrievers to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick runs",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repo root for ColBERT / hybrid imports",
    )
    parser.add_argument(
        "--colbert-index-root",
        default="colbert_data",
        help="Index root used when building the ColBERT indexes",
    )
    parser.add_argument(
        "--colbert-experiment",
        default="phase3_colbert",
        help="ColBERT experiment name used for both indexes",
    )
    parser.add_argument(
        "--zero-index-name",
        default="phase3_colbert_zero",
    )
    parser.add_argument(
        "--ft-index-name",
        default="phase3_colbert_ft",
    )
    args = parser.parse_args()

    split_path = Path(args.split).resolve()
    output_dir = Path(args.output_dir).resolve()
    project_root = Path(args.project_root).resolve()
    colbert_index_root = Path(args.colbert_index_root).resolve()

    samples = load_jsonl(split_path)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    doc_map_path = project_root / "colbert_data" / "data" / "doc_map.json"
    doc_map = load_doc_map(doc_map_path)
    indexed_docs = build_indexed_docs(doc_map)

    summaries: List[RetrieverSummary] = []

    if "hybrid" in args.retrievers:
        def hybrid_retrieve(query: str, k: int) -> List[str]:
            candidates = build_hybrid_backend()(query, k)
            return [candidate_module_id(c) for c in candidates if candidate_module_id(c)]

        hybrid_results, hybrid_summary = evaluate_retriever("hybrid", samples, hybrid_retrieve)
        write_outputs(output_dir / "hybrid", "hybrid", hybrid_results, hybrid_summary)
        print_summary(hybrid_summary)
        summaries.append(hybrid_summary)

    if "colbert_zero" in args.retrievers:
        zero_backend = build_colbert_backend(
            project_root=colbert_index_root,
            experiment=args.colbert_experiment,
            index_name=args.zero_index_name,
        )

        def colbert_zero_retrieve(query: str, k: int) -> List[str]:
            results = zero_backend(query, k)
            module_ids: List[str] = []
            for item in results:
                pid = int(item["pid"])
                if 0 <= pid < len(indexed_docs):
                    module_ids.append(canonical_id(indexed_docs[pid]["module"]))
            return module_ids

        zero_results, zero_summary = evaluate_retriever("colbert_zero", samples, colbert_zero_retrieve)
        write_outputs(output_dir / "colbert_zero", "colbert_zero", zero_results, zero_summary)
        print_summary(zero_summary)
        summaries.append(zero_summary)

    if "colbert_ft" in args.retrievers:
        ft_backend = build_colbert_backend(
            project_root=colbert_index_root,
            experiment=args.colbert_experiment,
            index_name=args.ft_index_name,
        )

        def colbert_ft_retrieve(query: str, k: int) -> List[str]:
            results = ft_backend(query, k)
            module_ids: List[str] = []
            for item in results:
                pid = int(item["pid"])
                if 0 <= pid < len(indexed_docs):
                    module_ids.append(canonical_id(indexed_docs[pid]["module"]))
            return module_ids

        ft_results, ft_summary = evaluate_retriever("colbert_ft", samples, colbert_ft_retrieve)
        write_outputs(output_dir / "colbert_ft", "colbert_ft", ft_results, ft_summary)
        print_summary(ft_summary)
        summaries.append(ft_summary)

    if summaries:
        print("\n" + "=" * 100)
        print("COMPARISON TABLE")
        print("=" * 100)
        header = (
            f"{'Retriever':<16}"
            f"{'Top1':>10}"
            f"{'Top5':>10}"
            f"{'Top10':>10}"
            f"{'MRR':>10}"
            f"{'nDCG@10':>12}"
            f"{'Avg ms':>12}"
        )
        print(header)
        print("-" * len(header))
        for s in summaries:
            print(
                f"{s.retriever:<16}"
                f"{s.top1:>10.4f}"
                f"{s.top5:>10.4f}"
                f"{s.top10:>10.4f}"
                f"{s.mrr:>10.4f}"
                f"{s.ndcg10:>12.4f}"
                f"{s.avg_retrieval_ms:>12.2f}"
            )


if __name__ == "__main__":
    main()