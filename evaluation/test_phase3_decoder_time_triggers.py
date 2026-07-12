from __future__ import annotations

import json
import yaml

from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.parser_state_decoder import generate_parser_state_yaml


QUERIES = [
    "create a virtual network called prod-vnet in resource group prod-rg",
    "delete subnet backend-subnet from virtual network prod-vnet resource group prod-rg",
    "create a virtual network called prod-vnet in resource group prod-rg with dns servers 1.1.1.1 and 8.8.8.8",
    "create a virtual network called prod-vnet in resource group prod-rg with tags env prod team platform",
    "create a storage account named mystorage in resource group prod-rg with https only true",
    "show details of virtual network prod-vnet",
    "list subnets in virtual network prod-vnet",
    "show details of virtual network gateway gateway1",
]


def _module_fqn_from_doc(doc: dict) -> str:
    collection = doc.get("collection", "")
    module = doc.get("module", "")
    return f"{collection}.{module}".strip(".")


def main() -> None:
    for query in QUERIES:
        print("=" * 100)
        print(query)

        doc_path = select_document(query)
        with open(doc_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        schema = extract_schema(doc_path)
        module_fqn = _module_fqn_from_doc(doc)

        out, meta = generate_parser_state_yaml(
            query=query,
            schema=schema,
            module_fqn=module_fqn,
            max_tokens=256,
            debug=False,
            return_metadata=True,
        )

        print("\nDOCUMENT:")
        print(doc_path)
        print("\nMETADATA:")
        print(meta)
        print("\nPARSER LOG:")
        for item in meta["parser_log"]:
            print(item)
        print("\nDECODER LOG:")
        for item in meta["decoder_log"]:
            print(item)
        print("\nTRIGGER LOG:")
        for item in meta["trigger_log"]:
            print(item)
        print("\nOUTPUT:")
        print(out)

        parsed = yaml.safe_load(out)
        assert parsed is not None, "YAML did not parse"

        assert meta["trigger_log"], "No trigger events recorded"
        assert meta["decoder_log"], "No decoder events recorded"
        assert meta["parser_log"], "No parser events recorded"
        assert "fire:header_emitted" in meta["trigger_log"], "Header trigger missing"

        print("PASS")


if __name__ == "__main__":
    main()