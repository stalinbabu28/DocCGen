from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

EMBEDDINGS_FILE = os.getenv(
    "ANSIBLE_EMBEDDINGS_FILE",
    str(Path(__file__).resolve().parent.parent / "ansible_embeddings.pkl"),
)

if os.path.exists(EMBEDDINGS_FILE):
    with open(EMBEDDINGS_FILE, "rb") as f:
        database = pickle.load(f)
else:
    database = []


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query, k=5):
    query_emb = model.encode(query)
    scores = []

    for item in database:
        score = cosine_similarity(query_emb, item["embedding"])
        scores.append((score, item["path"]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]