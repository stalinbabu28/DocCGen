from __future__ import annotations

import json
import random
import time
import yaml

from evaluation.test_cases_50 import TESTS
from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.dynamic_decoder import generate_dynamic_yaml


def _module_fqn_from_doc(doc: dict) -> str:
    collection = doc.get("collection", "")
    module = doc.get("module", "")
    return f"{collection}.{module}".strip(".")


def main() -> None:
    random.seed(0)
    iterations = 100
    failures = 0
    start = time.time()

    queries = [sample["query"] for sample in TESTS]

    for i in range(iterations):
        query = random.choice(queries)

        try:
            doc_path = select_document(query)
            with open(doc_path, "r", encoding="utf-8") as f:
                doc = json.load(f)

            schema = extract_schema(doc_path)
            module_fqn = _module_fqn_from_doc(doc)

            out = generate_dynamic_yaml(
                query=query,
                schema=schema,
                module_fqn=module_fqn,
                max_tokens=256,
                debug=False,
                return_metadata=False,
            )

            yaml.safe_load(out)
            print(f"[{i + 1}/{iterations}] OK")

        except Exception as exc:
            failures += 1
            print(f"[{i + 1}/{iterations}] FAIL: {exc}")

    elapsed = time.time() - start

    print("\n" + "=" * 100)
    print("STRESS TEST COMPLETE")
    print("ITERATIONS:", iterations)
    print("FAILURES:", failures)
    print("ELAPSED_SECONDS:", round(elapsed, 2))


if __name__ == "__main__":
    main()