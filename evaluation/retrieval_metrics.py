from __future__ import annotations

import math
from typing import Mapping, Sequence, Dict, List


def canonical_id(value: object) -> str:
    """
    Normalize module/library identifiers so comparisons are stable.
    Works for:
    - azure_rm_virtualnetwork
    - azure.azcollection.azure_rm_virtualnetwork
    - /path/to/azure.azcollection.azure_rm_virtualnetwork.json
    """
    s = str(value).strip()

    if "/" in s:
        s = s.rsplit("/", 1)[-1]

    for suffix in (".json", ".yml", ".yaml"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]

    s = s.replace("azure.azcollection.", "")
    return s


def hits_at_k(retrieved: Sequence[object], gold: Sequence[object], k: int) -> float:
    gold_set = {canonical_id(g) for g in gold}
    if not gold_set:
        return 0.0

    for item in retrieved[:k]:
        if canonical_id(item) in gold_set:
            return 1.0
    return 0.0


def recall_at_k(retrieved: Sequence[object], gold: Sequence[object], k: int) -> float:
    gold_set = {canonical_id(g) for g in gold}
    if not gold_set:
        return 0.0

    retrieved_set = {canonical_id(x) for x in retrieved[:k]}
    return len(retrieved_set & gold_set) / len(gold_set)


def precision_at_k(retrieved: Sequence[object], gold: Sequence[object], k: int) -> float:
    gold_set = {canonical_id(g) for g in gold}
    if not gold_set:
        return 0.0

    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0

    retrieved_set = {canonical_id(x) for x in retrieved_k}
    return len(retrieved_set & gold_set) / len(retrieved_k)


def reciprocal_rank(retrieved: Sequence[object], gold: Sequence[object], k: int | None = None) -> float:
    gold_set = {canonical_id(g) for g in gold}
    if not gold_set:
        return 0.0

    items = retrieved if k is None else retrieved[:k]
    for idx, item in enumerate(items, start=1):
        if canonical_id(item) in gold_set:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(rrs: Sequence[float]) -> float:
    return sum(rrs) / len(rrs) if rrs else 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        score += float(rel) / math.log2(i + 1)
    return score


def ndcg_at_k(
    retrieved: Sequence[object],
    gold_relevance: Mapping[object, float],
    k: int,
) -> float:
    if not gold_relevance:
        return 0.0

    retrieved_rels = [
        float(gold_relevance.get(canonical_id(item), 0.0))
        for item in retrieved[:k]
    ]
    dcg = dcg_at_k(retrieved_rels, k)

    ideal_rels = sorted((float(v) for v in gold_relevance.values()), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg