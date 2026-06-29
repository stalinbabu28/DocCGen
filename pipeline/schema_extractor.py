from __future__ import annotations

import json
from typing import Any, Dict


def _join_desc(desc: Any) -> str:
    if isinstance(desc, list):
        return " ".join(str(x).strip() for x in desc if str(x).strip())
    if desc is None:
        return ""
    return str(desc).strip()


def _normalize_type(raw: Any) -> str:
    t = str(raw or "str").lower().strip()
    aliases = {
        "str": "str",
        "string": "str",
        "path": "path",
        "bool": "bool",
        "boolean": "bool",
        "int": "int",
        "integer": "int",
        "float": "float",
        "number": "float",
        "list": "list",
        "array": "list",
        "dict": "dict",
        "mapping": "dict",
        "object": "dict",
    }
    return aliases.get(t, t)


def _extract_option(name: str, info: Dict[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "name": name,
        "required": bool(info.get("required", False)),
        "type": _normalize_type(info.get("type", "str")),
        "choices": list(info.get("choices", [])) if info.get("choices") else [],
        "default": info.get("default"),
        "description": _join_desc(info.get("description", "")),
        "aliases": list(info.get("aliases", [])) if info.get("aliases") else [],
        "suboptions": {},
        "elements": None,
    }

    nested = None
    if isinstance(info.get("suboptions"), dict):
        nested = info["suboptions"]
    elif isinstance(info.get("options"), dict) and spec["type"] == "dict":
        nested = info["options"]

    if isinstance(nested, dict):
        spec["suboptions"] = {
            child_name: _extract_option(child_name, child_info)
            for child_name, child_info in nested.items()
            if isinstance(child_info, dict)
        }

    elements = info.get("elements")
    if isinstance(elements, dict):
        spec["elements"] = _extract_option(f"{name}_item", elements)
    elif elements is not None:
        spec["elements"] = {"type": _normalize_type(elements)}

    return spec


def extract_schema(doc_file: str) -> Dict[str, Any]:
    with open(doc_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    schema: Dict[str, Any] = {
        "module": data.get("module", ""),
        "required": [],
        "optional": [],
        "choices": {},
        "types": {},
        "descriptions": {},
        "defaults": {},
        "aliases": {},
        "suboptions": {},
        "elements": {},
        "dependencies": {},
    }

    for name, info in data.get("options", {}).items():
        if not isinstance(info, dict):
            continue

        spec = _extract_option(name, info)

        if spec["required"]:
            schema["required"].append(name)
        else:
            schema["optional"].append(name)

        schema["types"][name] = spec["type"]
        schema["descriptions"][name] = spec["description"]

        if spec["choices"]:
            schema["choices"][name] = spec["choices"]

        if spec["default"] is not None:
            schema["defaults"][name] = spec["default"]

        if spec["aliases"]:
            schema["aliases"][name] = spec["aliases"]

        if spec["suboptions"]:
            schema["suboptions"][name] = spec["suboptions"]

        if spec["elements"] is not None:
            schema["elements"][name] = spec["elements"]

        deps = set()
        for key in ("required_if", "required_together", "mutually_exclusive"):
            value = info.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        deps.add(item)
                    elif isinstance(item, (list, tuple)):
                        deps.update(x for x in item if isinstance(x, str))
                    elif isinstance(item, dict):
                        deps.update(k for k in item.keys() if isinstance(k, str))

        if deps:
            schema["dependencies"][name] = sorted(deps)

    return schema


if __name__ == "__main__":
    schema = extract_schema("/home/dao-lab/stalin/azure_docs/azure.azcollection.azure_rm_virtualnetwork.json")
    from pprint import pprint
    pprint(schema)