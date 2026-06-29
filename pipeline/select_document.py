from pipeline.select_module_candidates import get_ranked_candidates
from pipeline.choose_module_llama import choose_module


def select_document(query: str) -> str:
    candidates = get_ranked_candidates(query, k=20)
    if not candidates:
        raise ValueError(f"No document found for query: {query}")

    chosen = choose_module(query, candidates)
    return chosen["path"]


def debug_document_selection(query: str):
    candidates = get_ranked_candidates(query, k=20)
    chosen = choose_module(query, candidates)

    print()
    print("=" * 80)
    print(query)

    for cand in candidates[:20]:
        print(round(cand["score"], 4), cand["path"])

    print()
    print("SELECTED:", chosen["module_fqn"])
    print("PATH:", chosen["path"])

    return candidates, chosen