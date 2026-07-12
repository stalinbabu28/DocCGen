from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pipeline.schema_text_utils import STOPWORDS

DOC_DIR = os.getenv("ANSIBLE_DOCS_DIR", str(Path(__file__).resolve().parent.parent / "docs"))

# Cap how much a single repeated token can contribute. Without this, a long,
# verbose module description that happens to repeat a common query word many
# times (e.g. "from", "to", "template") can outscore a short, exactly-correct
# module purely on word count, unrelated to actual relevance.
MAX_TOKEN_MATCH_CONTRIBUTION = 2


def _tokenize(text: str):
    tokens = re.findall(r"[a-z0-9_./-]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def retrieve_topk(query, k=20):
    query_tokens = _tokenize(query)
    compact_query = re.sub(r"[^a-z0-9]+", "", query.lower())

    scores = []

    for path in Path(DOC_DIR).rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        module = data.get("module", "")
        short_desc = data.get("short_description", "")
        description = " ".join(data.get("description", []) or [])

        text = f"{module} {short_desc} {description}"
        text_lower = text.lower()
        module_lower = module.lower()
        compact_module = re.sub(r"[^a-z0-9]+", "", module_lower)

        score = 0.0
        for token in query_tokens:
            score += min(text_lower.count(token), MAX_TOKEN_MATCH_CONTRIBUTION)
            if token in module_lower:
                score += 3.0

        if compact_query and compact_query in compact_module:
            score += 12.0

        score -= 0.05 * len(module_lower.split("_"))
        scores.append((float(score), str(path)))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]