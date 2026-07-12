#!/usr/bin/env python3
"""
Build a clean ColBERT collection TSV from Ansible module JSON docs.

Design goals:
- one passage per module by default
- keep the most retrieval-useful text only
- exclude repetitive boilerplate (notes, env-var caveats, filenames, version noise)
- preserve module name, short description, options, aliases, types, required/default/choices,
  nested suboptions, and examples when present

Input:
- files.txt: a list of JSON filenames relative to docs-root
- docs-root: directory containing the JSON files

Output:
- collection.tsv: one line per passage, format "pid<TAB>text"
- doc_map.json: optional mapping from pid to source file/module
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

BOILERPLATE_RE = re.compile(
    r"(environment variables|deprecated and will be removed|see the aws documentation|"
    r"see the amazon aws documentation|see the oci|see also|for more information|"
    r"host context|controller context|configuration file|config file|credentials?)",
    re.IGNORECASE,
)

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = URL_RE.sub("", text)
    text = text.replace("C(", "").replace(")", "")
    text = text.replace("I(", "").replace(")", "")
    text = text.replace("M(", "").replace(")", "")
    text = text.replace("L(", "").replace(")", "")
    text = text.replace("U(", "").replace(")", "")
    text = text.replace("P(", "").replace(")", "")
    text = text.replace("T(", "").replace(")", "")
    text = WS_RE.sub(" ", text).strip()
    return text


def is_boilerplate(text: str) -> bool:
    return bool(BOILERPLATE_RE.search(text))


def uniq_extend(out: List[str], seen: set[str], item: str) -> None:
    item = clean_text(item)
    if not item:
        return
    if item in seen:
        return
    seen.add(item)
    out.append(item)


def first_sentences(text: str, limit: int = 2) -> str:
    text = clean_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:limit]).strip()


def flatten_list(values: Any) -> List[str]:
    out: List[str] = []
    if isinstance(values, list):
        for v in values:
            if isinstance(v, str):
                t = clean_text(v)
                if t:
                    out.append(t)
            elif isinstance(v, dict):
                for k, vv in v.items():
                    t = clean_text(f"{k}: {vv}")
                    if t:
                        out.append(t)
            else:
                t = clean_text(str(v))
                if t:
                    out.append(t)
    elif isinstance(values, dict):
        for k, v in values.items():
            t = clean_text(f"{k}: {v}")
            if t:
                out.append(t)
    elif values is not None:
        t = clean_text(str(values))
        if t:
            out.append(t)
    return out


def add_option_section(lines: List[str], seen: set[str], name: str, spec: Dict[str, Any], prefix: str = "option") -> None:
    type_ = spec.get("type")
    aliases = spec.get("aliases") or []
    default = spec.get("default")
    required = spec.get("required")
    choices = spec.get("choices") or []
    desc = spec.get("description") or []
    elements = spec.get("elements")
    suboptions = spec.get("suboptions") or {}

    uniq_extend(lines, seen, f"{prefix}: {name}")
    if type_:
        uniq_extend(lines, seen, f"type: {type_}")
    if required is True:
        uniq_extend(lines, seen, "required: true")
    elif required is False:
        uniq_extend(lines, seen, "required: false")
    if default is not None:
        uniq_extend(lines, seen, f"default: {default}")
    if aliases:
        uniq_extend(lines, seen, "aliases: " + ", ".join(str(a) for a in aliases))
    if choices:
        uniq_extend(lines, seen, "choices: " + ", ".join(str(c) for c in choices))
    if elements:
        uniq_extend(lines, seen, f"elements: {elements}")

    for d in flatten_list(desc):
        if not is_boilerplate(d):
            uniq_extend(lines, seen, f"description: {first_sentences(d, 1)}")

    if isinstance(suboptions, dict) and suboptions:
        for sub_name, sub_spec in suboptions.items():
            if not isinstance(sub_spec, dict):
                continue
            add_option_section(lines, seen, f"{name}.{sub_name}", sub_spec, prefix="suboption")


def build_passage(module_path: Path, max_chars: int = 12000) -> Tuple[str, Dict[str, Any]]:
    data = json.loads(module_path.read_text(encoding="utf-8"))
    module = data.get("module") or module_path.stem
    collection = data.get("collection") or module_path.name.split(".", 1)[0]
    short_description = clean_text(data.get("short_description") or "")
    descriptions = flatten_list(data.get("description"))
    author = data.get("author")
    options = data.get("options") or {}

    lines: List[str] = []
    seen: set[str] = set()

    uniq_extend(lines, seen, f"module: {module}")
    uniq_extend(lines, seen, f"collection: {collection}")
    if short_description:
        uniq_extend(lines, seen, f"short_description: {short_description}")
    if author:
        for a in flatten_list(author):
            uniq_extend(lines, seen, f"author: {a}")

    for d in descriptions[:4]:
        if not is_boilerplate(d):
            uniq_extend(lines, seen, f"description: {first_sentences(d, 2)}")

    if isinstance(options, dict):
        for opt_name, opt_spec in options.items():
            if not isinstance(opt_spec, dict):
                continue
            add_option_section(lines, seen, opt_name, opt_spec)

    requirements = flatten_list(data.get("requirements"))
    for req in requirements[:5]:
        if not is_boilerplate(req):
            uniq_extend(lines, seen, f"requirement: {req}")

    # Avoid overly large passages while keeping the main doc coherent.
    passage = "\n".join(lines)
    if len(passage) > max_chars:
        passage = passage[:max_chars].rsplit("\n", 1)[0].strip()

    meta = {
        "module": module,
        "collection": collection,
        "source": str(module_path),
        "short_description": short_description,
    }
    return passage, meta


def iter_input_files(files_txt: Path, docs_root: Path) -> Iterable[Path]:
    for line in files_txt.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or not name.endswith(".json"):
            continue
        path = docs_root / name
        if path.exists():
            yield path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files-txt", required=True, type=Path)
    ap.add_argument("--docs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--map-out", required=True, type=Path)
    ap.add_argument("--max-chars", type=int, default=12000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.map_out.parent.mkdir(parents=True, exist_ok=True)

    doc_map: List[Dict[str, Any]] = []
    count = 0

    with args.out.open("w", encoding="utf-8") as out_f:
        for path in iter_input_files(args.files_txt, args.docs_root):
            try:
                passage, meta = build_passage(path, max_chars=args.max_chars)
            except Exception as e:
                print(f"[skip] {path.name}: {e}")
                continue

            if not passage.strip():
                continue

            pid = path.stem
            out_f.write(f"{pid}\t{passage.replace(chr(9), ' ')}\n")
            doc_map.append({"pid": pid, **meta})
            count += 1

    args.map_out.write_text(json.dumps(doc_map, indent=2), encoding="utf-8")
    print(f"Wrote {count} passages to {args.out}")
    print(f"Wrote map to {args.map_out}")


if __name__ == "__main__":
    main()
