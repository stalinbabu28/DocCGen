from pathlib import Path

from retrieval.colbert_retriever import ColBERTRetriever

PROJECT_ROOT = Path(__file__).resolve().parent.parent

retriever = ColBERTRetriever(PROJECT_ROOT)

results = retriever.retrieve(
    "create an ec2 instance",
    top_k=5,
)

for r in results:
    print("=" * 80)
    print(r["rank"])
    print(r["module"])
    print(r["score"])
    print(r["text"][:500])