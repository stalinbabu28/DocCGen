import json
import os
import re

DOC_DIR = "/home/dao-lab/stalin/azure_docs"


def _tokenize(text: str):
    return re.findall(r"[a-z0-9_./-]+", text.lower())


def retrieve_topk(query, k=20):
    """
    Sparse retriever used as one signal in the shortlist.
    Returns [(score, path), ...].
    """
    query_tokens = _tokenize(query)
    compact_query = re.sub(r"[^a-z0-9]+", "", query.lower())

    scores = []

    for file in os.listdir(DOC_DIR):
        if not file.endswith(".json"):
            continue

        path = os.path.join(DOC_DIR, file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        module = data.get("module", "")
        short_desc = data.get("short_description", "")
        description = " ".join(data.get("description", []))

        text = f"{module} {short_desc} {description}"
        text_lower = text.lower()

        module_lower = module.lower()
        compact_module = re.sub(r"[^a-z0-9]+", "", module_lower)

        score = 0.0

        for token in query_tokens:
            score += text_lower.count(token)

            if token in module_lower:
                score += 3.0

        if compact_query and compact_query in compact_module:
            score += 12.0

        # Prefer shorter, more exact module names over long siblings
        score -= 0.05 * len(module_lower.split("_"))

        scores.append((float(score), path))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]