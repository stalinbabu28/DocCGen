from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from evaluation.test_cases_50 import TESTS
from evaluation.retrieval_metrics import (
    canonical_id,
    hits_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from pipeline.select_module_candidates import get_ranked_candidates

RETRIEVAL_K = 20
METRIC_KS = (1, 3, 5)


@dataclass
class QueryRetrievalResult:
    query: str
    gold: str
    retrieved: List[str]
    rr: float
    hits: Dict[int, float]
    recalls: Dict[int, float]
    precisions: Dict[int, float]
    ndcg_3: float
    ndcg_5: float


def _candidate_module_id(candidate: dict) -> str:
    module = candidate.get("module")
    if module:
        return canonical_id(module)

    module_fqn = candidate.get("module_fqn", "")
    return canonical_id(module_fqn.split(".")[-1])


def _gold_relevance_map(gold_module: str) -> Dict[str, float]:
    return {canonical_id(gold_module): 1.0}


def evaluate_query(query: str, gold_module: str, k: int = RETRIEVAL_K) -> QueryRetrievalResult:
    candidates = get_ranked_candidates(query, k=k)
    retrieved_ids = [_candidate_module_id(c) for c in candidates]

    gold = canonical_id(gold_module)
    gold_set = [gold]

    hits: Dict[int, float] = {}
    recalls: Dict[int, float] = {}
    precisions: Dict[int, float] = {}

    for kk in METRIC_KS:
        hits[kk] = hits_at_k(retrieved_ids, gold_set, kk)
        recalls[kk] = recall_at_k(retrieved_ids, gold_set, kk)
        precisions[kk] = precision_at_k(retrieved_ids, gold_set, kk)

    rr = reciprocal_rank(retrieved_ids, gold_set, k=k)
    ndcg_3 = ndcg_at_k(retrieved_ids, _gold_relevance_map(gold_module), 3)
    ndcg_5 = ndcg_at_k(retrieved_ids, _gold_relevance_map(gold_module), 5)

    return QueryRetrievalResult(
        query=query,
        gold=gold,
        retrieved=retrieved_ids,
        rr=rr,
        hits=hits,
        recalls=recalls,
        precisions=precisions,
        ndcg_3=ndcg_3,
        ndcg_5=ndcg_5,
    )


def main() -> None:
    results: List[QueryRetrievalResult] = []

    hit_totals = {k: 0.0 for k in METRIC_KS}
    recall_totals = {k: 0.0 for k in METRIC_KS}
    precision_totals = {k: 0.0 for k in METRIC_KS}
    rrs: List[float] = []
    ndcg3s: List[float] = []
    ndcg5s: List[float] = []

    for sample in TESTS:
        query = sample["query"]
        gold_module = sample["expected_module"]
        result = evaluate_query(query, gold_module, k=RETRIEVAL_K)
        results.append(result)

        for k in METRIC_KS:
            hit_totals[k] += result.hits[k]
            recall_totals[k] += result.recalls[k]
            precision_totals[k] += result.precisions[k]

        rrs.append(result.rr)
        ndcg3s.append(result.ndcg_3)
        ndcg5s.append(result.ndcg_5)

        print("\n" + "=" * 100)
        print(f"Query: {query}")
        print(f"Gold:  {gold_module}")

        print("\nTop retrieved:")
        candidates = get_ranked_candidates(query, k=min(RETRIEVAL_K, 10))
        for idx, cand in enumerate(candidates[:10], start=1):
            mid = _candidate_module_id(cand)
            score = cand.get("final_score", cand.get("score", 0.0))
            mark = " <-- GOLD" if canonical_id(mid) == canonical_id(gold_module) else ""
            print(f"{idx:>2}. {mid:<40} score={score:.4f}{mark}")

        print("\nPer-query metrics:")
        for k in METRIC_KS:
            print(
                f"Hits@{k}={result.hits[k]:.0f}  "
                f"Recall@{k}={result.recalls[k]:.4f}  "
                f"Precision@{k}={result.precisions[k]:.4f}"
            )
        print(f"RR={result.rr:.4f}")
        print(f"nDCG@3={result.ndcg_3:.4f}")
        print(f"nDCG@5={result.ndcg_5:.4f}")

    n = len(results) if results else 1

    print("\n" + "=" * 100)
    print("RETRIEVAL RESULTS")
    print("=" * 100)

    for k in METRIC_KS:
        print(f"Hits@{k}:      {hit_totals[k] / n:.4f}")
    for k in METRIC_KS:
        print(f"Recall@{k}:    {recall_totals[k] / n:.4f}")
    for k in METRIC_KS:
        print(f"Precision@{k}: {precision_totals[k] / n:.4f}")

    print(f"MRR:          {mean_reciprocal_rank(rrs):.4f}")
    print(f"nDCG@3:       {sum(ndcg3s) / n:.4f}")
    print(f"nDCG@5:       {sum(ndcg5s) / n:.4f}")

    print("\nNote:")
    print("With one gold module per query, Recall@k and Hits@k are identical.")
    print("These metrics evaluate retrieval only, not YAML generation.")


if __name__ == "__main__":
    main()