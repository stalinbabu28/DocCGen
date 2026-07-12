from __future__ import annotations

import re
from pathlib import Path

from retrieval.intent_classifier import classify_intent


def _module_short_from_path(path: str) -> str:
    base = Path(path).name
    if base.endswith(".json"):
        base = base[:-5]
    return base.split(".")[-1].lower()


def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def rerank(query, results):
    intent = classify_intent(query)
    query_lower = query.lower()
    query_tokens = set(_tokenize(query))
    reranked = []

    for score, path in results:
        adjusted_score = float(score)
        module_short = _module_short_from_path(path)
        module_tokens = set(
            _tokenize(
                module_short.replace("_info", "").replace("_facts", "").replace("_", " ")
            )
        )

        is_info_module = module_short.endswith("_info") or module_short.endswith("_facts")

        if intent == "resource" and is_info_module:
            adjusted_score -= 0.10
        elif intent == "info" and is_info_module:
            adjusted_score += 0.10

        # Ratio, not raw count: a long, verbose module name (e.g.
        # "na_ontap_wait_for_condition") can share several literal words with
        # a generic query purely by being wordy, while a short exact-match
        # module (e.g. "wait_for") only needs to match once. Normalizing by
        # how much of the module's own name is covered keeps precise short
        # names from losing to incidentally-verbose ones.
        overlap = query_tokens & module_tokens
        if module_tokens:
            adjusted_score += (len(overlap) / len(module_tokens)) * 0.15

        if any(x in module_tokens for x in {"instance", "extension", "link", "group"}):
            adjusted_score -= 0.03

        if "devtestlab" in module_short and "devtestlab" not in query_lower:
            adjusted_score -= 0.05

        reranked.append((adjusted_score, path))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked