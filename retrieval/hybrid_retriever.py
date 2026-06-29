import numpy as np

from retrieval.embedding_retriever import retrieve
from retrieval.retrieve_topk import retrieve_topk


def normalize(results):

    if not results:
        return {}

    scores = [s for s, _ in results]

    mn = min(scores)
    mx = max(scores)

    if mx == mn:

        return {
            path: 1.0
            for score, path in results
        }

    normalized = {}

    for score, path in results:

        normalized[path] = (
            (score - mn)
            / (mx - mn)
        )

    return normalized


def hybrid_retrieve(
    query,
    k=20
):

    dense = retrieve(query, k=50)

    sparse = retrieve_topk(query, k=50)

    dense_scores = normalize(dense)

    sparse_scores = normalize(sparse)

    all_paths = set(
        dense_scores.keys()
    ) | set(
        sparse_scores.keys()
    )

    combined = []

    for path in all_paths:

        dense_score = dense_scores.get(
            path,
            0
        )

        sparse_score = sparse_scores.get(
            path,
            0
        )

        final_score = (
            0.7 * dense_score
            +
            0.3 * sparse_score
        )

        combined.append(
            (
                final_score,
                path
            )
        )

    combined.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return combined[:k]