from __future__ import annotations

import json

from llama_cpp import LlamaGrammar

from evaluation.test_cases_50 import TESTS
from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.dynamic_decoder import StagewiseDynamicDecoder


def _module_fqn_from_doc(doc: dict) -> str:
    collection = doc.get("collection", "")
    module = doc.get("module", "")
    return f"{collection}.{module}".strip(".")


def main() -> None:
    decoder = StagewiseDynamicDecoder()
    total_pieces = 0
    failures = 0

    for i, sample in enumerate(TESTS, start=1):
        query = sample["query"]

        doc_path = select_document(query)
        with open(doc_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        schema = extract_schema(doc_path)
        module_fqn = _module_fqn_from_doc(doc)

        state = decoder.plan_pieces(query=query, schema=schema, module_fqn=module_fqn)

        print("\n" + "=" * 100)
        print(f"[{i}/{len(TESTS)}] {query}")
        print("PIECES:", [p.label for p in state.pieces])

        for piece in state.pieces:
            total_pieces += 1
            try:
                LlamaGrammar.from_string(piece.grammar_text)
            except Exception as exc:
                failures += 1
                print("INVALID PIECE:", piece.label)
                print("ERROR:", exc)
                print("GRAMMAR:")
                print(piece.grammar_text)

    print("\n" + "=" * 100)
    print("TOTAL PIECES:", total_pieces)
    print("GRAMMAR FAILURES:", failures)


if __name__ == "__main__":
    main()