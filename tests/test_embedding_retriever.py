from embedding_retriever import retrieve

results = retrieve(
    "create an AKS cluster"
)

for score, path in results:
    print(round(score, 4), path)