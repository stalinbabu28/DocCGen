from __future__ import annotations

import json
import yaml

from llama_cpp import LlamaGrammar

from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.dynamic_decoder import StagewiseDynamicDecoder, generate_dynamic_yaml


QUERIES = [
    "create a virtual network called prod-vnet in resource group prod-rg",
    "delete subnet backend-subnet from virtual network prod-vnet resource group prod-rg",
    "create a virtual network called prod-vnet in resource group prod-rg with dns servers 1.1.1.1 and 8.8.8.8",
    "create a virtual network called prod-vnet in resource group prod-rg with tags env prod team platform",
    "create a storage account named mystorage in resource group prod-rg with https only true",
    "show details of virtual network prod-vnet",
    "list subnets in virtual network prod-vnet",
]


def _module_fqn_from_doc(doc: dict) -> str:
    collection = doc.get("collection", "")
    module = doc.get("module", "")
    return f"{collection}.{module}".strip(".")


def main() -> None:
    decoder = StagewiseDynamicDecoder()

    for query in QUERIES:
        print("=" * 100)
        print(query)

        doc_path = select_document(query)
        with open(doc_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        schema = extract_schema(doc_path)
        module_fqn = _module_fqn_from_doc(doc)

        state = decoder.plan_pieces(query=query, schema=schema, module_fqn=module_fqn)

        print("\nDOCUMENT:")
        print(doc_path)
        print("\nMODULE_FQN:")
        print(module_fqn)
        print("\nACTIVE_FIELDS:")
        print(state.active_fields)
        print("\nVALUE_HINTS:")
        print(state.value_hints)
        print("\nPIECES:")
        for p in state.pieces:
            print(" -", p.label)

        print("\nGRAMMAR VALIDATION:")
        for p in state.pieces:
            try:
                LlamaGrammar.from_string(p.grammar_text)
                print("OK:", p.label)
            except Exception as exc:
                print("BAD:", p.label, "ERROR:", exc)

        out, meta = generate_dynamic_yaml(
            query=query,
            schema=schema,
            module_fqn=module_fqn,
            max_tokens=256,
            debug=False,
            return_metadata=True,
        )

        print("\nMETADATA:")
        print(meta)
        print("\nOUTPUT:")
        print(out)

        try:
            parsed = yaml.safe_load(out)
            print("\nYAML PARSE: PASS" if parsed is not None else "\nYAML PARSE: FAIL (None)")
        except Exception as exc:
            print("\nYAML PARSE: FAIL")
            print(exc)


if __name__ == "__main__":
    main()