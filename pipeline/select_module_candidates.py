from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from retrieval.embedding_retriever import retrieve as dense_retrieve
from retrieval.rerank_results import rerank

try:
    from retrieval.retrieve_topk import retrieve_topk as sparse_retrieve
except Exception:
    sparse_retrieve = None

INFO_CUES = (
    " show ",
    " details ",
    " information ",
    " info ",
    " get ",
    " list ",
    " display ",
    " describe ",
)

RESOURCE_HINTS: List[Tuple[str, str]] = [
    ("virtual network gateway", "virtualnetworkgateway"),
    ("virtual network", "virtualnetwork"),
    ("network interface", "networkinterface"),
    ("dns zone", "dnszone"),
    ("public ip address", "publicipaddress"),
    ("route table", "routetable"),
    ("storage account", "storageaccount"),
    ("managed disk", "manageddisk"),
    ("load balancer", "loadbalancer"),
    ("application gateway", "appgateway"),
    ("key vault", "keyvault"),
    ("container registry", "containerregistry"),
    ("web app", "webapp"),
    ("aks cluster", "aks"),
    ("sql database", "sqldatabase"),
    ("sql server", "sqlserver"),
    ("subnet", "subnet"),
]


def _normalize(results: List[Tuple[float, str]]) -> Dict[str, float]:
    if not results:
        return {}

    scores = [score for score, _ in results]
    mn = min(scores)
    mx = max(scores)

    if mx == mn:
        return {path: 1.0 for _, path in results}

    return {
        path: (score - mn) / (mx - mn)
        for score, path in results
    }


@lru_cache(maxsize=2048)
def _load_doc(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _module_slug(candidate: dict) -> str:
    slug = candidate["module_fqn"].split(".")[-1].lower()
    for prefix in ("azure_rm_", "azure_"):
        if slug.startswith(prefix):
            slug = slug[len(prefix):]
    if slug.endswith("_info"):
        slug = slug[:-5]
    return slug


def _query_has_info_cue(query: str) -> bool:
    q = f" {query.lower()} "
    return any(cue in q for cue in INFO_CUES)


def _resource_hint_boost(query: str, candidate: dict) -> float:
    q = f" {query.lower()} "
    slug = _module_slug(candidate)
    score = float(candidate.get("score", 0.0))

    if _query_has_info_cue(query):
        if candidate["module_fqn"].endswith("_info"):
            score += 4.0
        else:
            score -= 1.5
    else:
        if candidate["module_fqn"].endswith("_info"):
            score -= 1.0

    for phrase, keyword in RESOURCE_HINTS:
        if phrase not in q:
            continue

        if keyword == "virtualnetwork":
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0
            elif "networkinterface" in slug:
                score -= 6.0
        elif keyword == "dnszone":
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0
            elif "dnszonegroup" in slug:
                score -= 4.0
        elif keyword == "keyvault":
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0
        else:
            if slug == keyword:
                score += 12.0
            elif slug.startswith(keyword):
                score += 6.0
            elif keyword in slug:
                score += 2.0

    return score


def get_ranked_candidates(
    query: str,
    k: int = 10,
    dense_k: int = 50,
    sparse_k: int = 50,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
):
    """
    Hybrid shortlist:
    - dense retrieval
    - sparse retrieval
    - reranking
    Returns a list of candidate dicts.
    """
    dense_results = dense_retrieve(query, k=dense_k)
    sparse_results = sparse_retrieve(query, k=sparse_k) if sparse_retrieve else []

    dense_map = _normalize(dense_results)
    sparse_map = _normalize(sparse_results)

    all_paths = set(dense_map.keys()) | set(sparse_map.keys())

    combined = []
    for path in all_paths:
        dense_score = dense_map.get(path, 0.0)
        sparse_score = sparse_map.get(path, 0.0)
        final_score = dense_weight * dense_score + sparse_weight * sparse_score
        combined.append((final_score, path))

    combined = rerank(query, combined)
    combined.sort(key=lambda x: x[0], reverse=True)

    candidates = []
    for score, path in combined[:k]:
        doc = _load_doc(path)
        module = doc.get("module", "")
        collection = doc.get("collection", "")
        module_fqn = f"{collection}.{module}" if collection else module

        candidate = {
            "score": score,
            "path": path,
            "module": module,
            "module_fqn": module_fqn,
            "collection": collection,
            "short_description": doc.get("short_description", ""),
        }
        candidate["heuristic_score"] = _resource_hint_boost(query, candidate)
        candidate["final_score"] = candidate["score"] + candidate["heuristic_score"]
        candidates.append(candidate)

    candidates.sort(key=lambda c: (c["final_score"], c["score"]), reverse=True)
    return candidates


def select_best_candidate(query: str, k: int = 10) -> Optional[dict]:
    candidates = get_ranked_candidates(query, k=k)
    return candidates[0] if candidates else None