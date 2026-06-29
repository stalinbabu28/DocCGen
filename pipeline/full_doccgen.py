import json
from typing import Any, Dict, Optional

import yaml

from pipeline.select_document import select_document
from pipeline.schema_extractor import extract_schema
from pipeline.extract_params_yaml import generate_constrained_yaml


ALIASES = {
    "azure.azcollection.azure_rm_keyvault": {"vault_name": "name"},
    "azure.azcollection.azure_rm_keyvault_info": {"vault_name": "name"},
}


def _canonicalize_params(module_fqn: str, params: Optional[Dict]) -> Optional[Dict]:
    if not params:
        return params

    out = dict(params)
    mapping = ALIASES.get(module_fqn, {})
    for src, dst in mapping.items():
        if src in out and dst not in out:
            out[dst] = out.pop(src)
    return out


def _extract_module_args(parsed_yaml: Any, module_fqn: str) -> Optional[Dict]:
    """
    Expected direct-YAML shape:
    - name: Generated Task
      azure.azcollection.azure_rm_xxx:
        key: value

    Return the inner module-argument dict.
    """
    if isinstance(parsed_yaml, list) and parsed_yaml:
        first = parsed_yaml[0]
        if isinstance(first, dict):
            if module_fqn in first and isinstance(first[module_fqn], dict):
                return _canonicalize_params(module_fqn, first[module_fqn])

            for key, value in first.items():
                if key != "name" and isinstance(value, dict):
                    return _canonicalize_params(module_fqn, value)

    if isinstance(parsed_yaml, dict):
        if module_fqn in parsed_yaml and isinstance(parsed_yaml[module_fqn], dict):
            return _canonicalize_params(module_fqn, parsed_yaml[module_fqn])

    return None


def run_pipeline(query: str, max_tokens: int = 256):
    doc_path = select_document(query)

    with open(doc_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    schema = extract_schema(doc_path)
    module_fqn = f"{doc.get('collection', '')}.{doc.get('module', '')}".strip(".")

    raw = generate_constrained_yaml(
        query=query,
        schema=schema,
        module_fqn=module_fqn,
        max_tokens=max_tokens,
    )

    try:
        parsed_yaml = yaml.safe_load(raw)
    except Exception:
        parsed_yaml = None

    params = _extract_module_args(parsed_yaml, module_fqn) if parsed_yaml is not None else None

    return {
        "document": doc_path,
        "module_name": doc.get("module", ""),
        "module_fqn": module_fqn,
        "schema": schema,
        "raw": raw,
        "params": params,
        "yaml": raw,
        "yaml_parsed": parsed_yaml,
    }


if __name__ == "__main__":
    query = input("Query: ")
    result = run_pipeline(query)

    print("\nDOCUMENT:")
    print(result["document"])

    print("\nMODULE:")
    print(result["module_fqn"])

    print("\nRAW:")
    print(result["raw"])

    print("\nPARAMS:")
    print(result["params"])

    print("\nYAML:")
    print(result["yaml"])