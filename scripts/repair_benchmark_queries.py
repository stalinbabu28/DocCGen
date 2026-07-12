from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}={fmt(val)}" for k, val in v.items())
    s = str(v).strip()
    return s.rstrip(".,;:!? ")


def module_short(case: Dict[str, Any]) -> str:
    return str(case.get("module", "")).strip().lower()


def build_expected_yaml(case: Dict[str, Any]) -> str:
    module_fqn = case["expected_module"]
    fields = case["expected_fields"] or {}
    lines = [f"- name: Generated Task", f"  {module_fqn}:"]
    for k, v in fields.items():
        if isinstance(v, list):
            lines.append(f"    {k}:")
            for item in v:
                lines.append(f"      - {fmt(item)}")
        elif isinstance(v, dict):
            lines.append(f"    {k}:")
            for kk, vv in v.items():
                lines.append(f"      {kk}: {fmt(vv)}")
        else:
            lines.append(f"    {k}: {fmt(v)}")
    return "\n".join(lines)


def build_query(case: Dict[str, Any]) -> str:
    module = module_short(case)
    fields = case["expected_fields"] or {}
    task_type = case.get("task_type", "create")

    name = fmt(fields.get("name", "demo-resource"))
    path = fmt(fields.get("path", "demo-path"))
    src = fmt(fields.get("src", "demo-src"))
    dest = fmt(fields.get("dest", "demo-dest"))
    url = fmt(fields.get("url", "demo-url"))
    line = fmt(fields.get("line", "demo-line"))
    rg = fmt(fields.get("resource_group", "prod-rg"))
    state = fmt(fields.get("state", "present"))

    # Task-specific templates
    if task_type in {"facts"} or module.endswith("_facts"):
        direct = f"Gather facts for {module}."
    elif task_type in {"info"} or module.endswith("_info"):
        if "name" in fields:
            direct = f"Show details of {module} {name}."
        elif "resource_group" in fields:
            direct = f"Show details of {module} in resource group {rg}."
        else:
            direct = f"Show details of {module}."
    elif task_type == "delete" or state == "absent":
        if "name" in fields and "resource_group" in fields:
            direct = f"Delete {module} named {name} in resource group {rg}."
        elif "name" in fields:
            direct = f"Delete {module} named {name}."
        else:
            direct = f"Remove {module}."
    else:
        if module == "template":
            direct = f"Render template from {src} to {dest}."
        elif module == "unarchive":
            direct = f"Extract archive from {src} into {dest}."
        elif module == "assemble":
            direct = f"Assemble files from {src} into {dest}."
        elif module == "copy":
            direct = f"Copy a file to {dest}."
        elif module == "get_url":
            direct = f"Download {url} to {dest}."
        elif module == "lineinfile":
            direct = f"Ensure a line is present in {path}."
        elif module == "file":
            direct = f"Create a file at {path}."
        elif module == "service":
            direct = f"Ensure service {name} is running."
        elif module == "cron":
            direct = f"Create cron job named {name}."
        elif module == "wait_for":
            direct = f"Wait until the condition is {state}."
        elif "resource_group" in fields and "name" in fields:
            direct = f"Create {module} named {name} in resource group {rg}."
        elif "name" in fields:
            direct = f"Create {module} named {name}."
        elif "path" in fields:
            direct = f"Create {module} at {path}."
        else:
            # Fallback: mention the actual required fields, not a fake name.
            clause = " and ".join(k.replace("_", " ") for k in list(fields.keys())[:3]) or module.replace("_", " ")
            direct = f"Create {clause} for {module}."

    return direct.rstrip(".")


def styles_from_direct(direct: str, module: str, fields: Dict[str, Any]) -> Dict[str, str]:
    subject = module.replace("_", " ")
    name = fmt(fields.get("name", "demo-resource"))
    path = fmt(fields.get("path", "demo-path"))

    return {
        "direct": direct,
        "verbose": f"Please generate {subject} for the requested configuration: {direct.lower()}",
        "search": f"How do I configure {subject} for this request: {direct.lower()}",
        "implicit": f"Set up {subject} according to the request. {direct}",
    }


def main():
    parser = argparse.ArgumentParser(description="Repair benchmark queries to be module-aware.")
    parser.add_argument("--input", required=True, help="Input benchmark JSONL")
    parser.add_argument("--output", required=True, help="Output repaired JSONL")
    parser.add_argument("--inplace", action="store_true", help="Overwrite input file")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    records = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    repaired = []
    for case in records:
        direct = build_query(case)
        case["query"] = direct
        case["query_styles"] = styles_from_direct(direct, module_short(case), case.get("expected_fields", {}))
        case["expected_yaml"] = build_expected_yaml(case)

        # Keep the original benchmark metadata intact.
        case["review_status"] = case.get("review_status", "draft")
        repaired.append(case)

    target = inp if args.inplace else out
    with target.open("w", encoding="utf-8") as f:
        for case in repaired:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"Wrote {len(repaired)} repaired cases to {target}")


if __name__ == "__main__":
    main()