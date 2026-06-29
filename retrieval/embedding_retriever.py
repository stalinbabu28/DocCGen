import pickle
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("azure_embeddings.pkl", "rb") as f:
    database = pickle.load(f)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(query, k=5):

    query_emb = model.encode(query)

    scores = []

    for item in database:

        score = cosine_similarity(
            query_emb,
            item["embedding"]
        )

        scores.append(
            (score, item["path"])
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scores[:k]