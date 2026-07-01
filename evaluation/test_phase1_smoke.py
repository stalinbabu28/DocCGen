from __future__ import annotations

import sys
import yaml

from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.dynamic_decoder import generate_dynamic_yaml

SMOKE_CASES = [
    "create a virtual network called prod-vnet in resource group prod-rg",
    "delete subnet backend-subnet from virtual network prod-vnet resource group prod-rg",
    "create a virtual network called prod-vnet in resource group prod-rg with dns servers 1.1.1.1 and 8.8.8.8",
    "create a virtual network called prod-vnet in resource group prod-rg with tags env prod team platform",
    "create a storage account named mystorage in resource group prod-rg with https only true",
    "create a storage account named mystorage in resource group prod-rg with allow blob public access false",
    "show details of virtual network prod-vnet",
    "list subnets in virtual network prod-vnet",
]


def _module_fqn_from_doc(doc: dict) -> str:
    collection = doc.get("collection", "")
    module = doc.get("module", "")
    return f"{collection}.{module}".strip(".")


def main() -> None:
    failures = 0

    for query in SMOKE_CASES:
        print("=" * 100)
        print(query)

        doc_path = select_document(query)
        with open(doc_path, "r", encoding="utf-8") as f:
            import json
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

        print("\nDOCUMENT:")
        print(doc_path)
        print("\nOUTPUT:")
        print(out)

        if not out.strip():
            print("FAIL: empty output")
            failures += 1
            continue

        try:
            parsed = yaml.safe_load(out)
        except Exception as exc:
            print("FAIL: YAML parse error:", exc)
            failures += 1
            continue

        if parsed is None:
            print("FAIL: parsed YAML is None")
            failures += 1
        else:
            print("PASS")

    print("\n" + "=" * 100)
    print("SMOKE TEST FAILURES:", failures)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()