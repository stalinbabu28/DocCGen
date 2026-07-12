from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from pipeline.parser_state_decoder import generate_parser_state_yaml
from pipeline.schema_extractor import extract_schema

STOP_FIELDS = {
    "access_key",
    "secret_key",
    "session_token",
    "validate_certs",
    "endpoint_url",
    "aws_ca_bundle",
    "aws_config",
    "profile",
    "region",
    "debug_botocore_endpoint_logs",
    "endpoint",
    "token",
    "password",
    "username",
    "api_key",
    "client_id",
    "client_secret",
    "certificate",
    "private_key",
}

SAFE_BOOLEAN_FIELDS = {
    "state",
    "enabled",
    "disable",
    "disabled",
    "validate",
    "check",
    "public",
    "private",
    "append_tags",
}

SAFE_LIST_FIELDS = {
    "trail_names",
    "dns_servers",
    "tags",
    "labels",
    "rules",
    "subnets",
    "networks",
    "items",
    "members",
    "targets",
    "servers",
    "addresses",
    "names",
}

SAFE_DICT_FIELDS = {
    "tags",
    "labels",
    "metadata",
    "annotations",
    "options",
    "settings",
    "parameters",
    "attributes",
    "properties",
    "args",
}


def _join_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip())
    if value is None:
        return ""
    return str(value).strip()


def _module_fqn_from_doc(doc: Dict[str, Any]) -> str:
    collection = doc.get("collection", "")
    module = doc.get("module", "")
    return f"{collection}.{module}".strip(".")


def _resolve_doc_path(line: str, docs_root: Path) -> Optional[Path]:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None

    p = Path(raw)
    if p.is_file():
        return p.resolve()

    candidate = docs_root / raw
    if candidate.is_file():
        return candidate.resolve()

    matches = list(docs_root.rglob(Path(raw).name))
    if matches:
        return matches[0].resolve()

    return None


def _load_paths(files_txt: Path, docs_root: Path) -> List[Path]:
    paths: List[Path] = []
    with files_txt.open("r", encoding="utf-8") as f:
        for line in f:
            path = _resolve_doc_path(line, docs_root)
            if path is not None:
                paths.append(path)
    return paths


def _normalize_field(field: str) -> str:
    return field.replace("_", " ").strip().lower()


def _field_clause(field: str, field_type: str) -> Optional[str]:
    f = field.lower()

    if f in STOP_FIELDS:
        return None

    if f == "name":
        return "called demo-item"
    if f == "state":
        return "delete"
    if f in {"resource_group", "resourcegroup", "group"}:
        return "in resource group demo-rg"
    if f in {"location", "region"}:
        return "in us-east-1"
    if f == "append_tags":
        return "with append tags true"
    if f in {"tags", "labels", "metadata", "annotations"}:
        return "with tags env prod team platform"
    if f in {"dns_servers", "servers", "addresses", "names", "items", "targets", "rules", "subnets", "networks"}:
        return f"with {_normalize_field(field)} alpha and beta"

    if field_type == "bool":
        return f"with {_normalize_field(field)} true"
    if field_type == "int":
        return f"with {_normalize_field(field)} 1"
    if field_type == "list":
        return f"with {_normalize_field(field)} alpha and beta"
    if field_type == "dict":
        return f"with {_normalize_field(field)} env prod team platform"

    return f"with {_normalize_field(field)} demo-value"


def _usable_fields(schema: Dict[str, Any]) -> List[str]:
    required = list(schema.get("required", []))
    optional = list(schema.get("optional", []))
    all_fields = []
    for f in required + optional:
        if f not in all_fields:
            all_fields.append(f)
    return [f for f in all_fields if f not in STOP_FIELDS]


def _build_probe_query(doc: Dict[str, Any], schema: Dict[str, Any], idx: int) -> str:
    short_desc = _join_text(doc.get("short_description", ""))
    description = _join_text(doc.get("description", []))
    module = str(doc.get("module", "")).strip()
    module_fqn = _module_fqn_from_doc(doc)

    subject = short_desc or description or module.replace("_", " ")
    subject = subject.strip().rstrip(".")

    is_infoish = (
        module_fqn.endswith("_info")
        or module_fqn.endswith("_facts")
        or not doc.get("has_action", True)
    )

    if is_infoish:
        base = f"show details of {subject}"
    else:
        if "state" in (schema.get("types", {}) or {}) and idx % 5 == 1:
            base = f"delete {subject}"
        else:
            base = f"create {subject}"

    fields = _usable_fields(schema)
    field_types = schema.get("types", {}) or {}

    cue_clauses: List[str] = []
    for field in fields:
        if len(cue_clauses) >= 2:
            break
        clause = _field_clause(field, str(field_types.get(field, "str")).lower())
        if clause:
            cue_clauses.append(clause)

    if cue_clauses and not is_infoish:
        base = f"{base} {' and '.join(cue_clauses)}"

    return re.sub(r"\s+", " ", base).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 over a corpus of docs listed in files.txt.")
    parser.add_argument("--files-txt", required=True, help="Path to files.txt (one filename per line).")
    parser.add_argument("--docs-root", required=True, help="Directory that contains the corpus JSON docs.")
    parser.add_argument("--output-dir", default="evaluation/phase3_corpus", help="Where to write results.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of docs.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Decoder max tokens.")
    parser.add_argument("--print-every", type=int, default=50, help="Progress print frequency. Use 0 to disable.")
    args = parser.parse_args()

    files_txt = Path(args.files_txt).expanduser().resolve()
    docs_root = Path(args.docs_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    doc_paths = _load_paths(files_txt, docs_root)
    if args.limit > 0:
        doc_paths = doc_paths[: args.limit]

    total = len(doc_paths)
    if total == 0:
        raise SystemExit(f"No docs found from {files_txt} using docs root {docs_root}")

    summary_path = output_dir / "summary.jsonl"
    failures_path = output_dir / "failures.jsonl"

    parsed_ok = 0
    parser_docs = 0
    decoder_docs = 0
    trigger_docs = 0
    total_switches = 0
    total_trigger_events = 0
    failures: List[Dict[str, Any]] = []

    with summary_path.open("w", encoding="utf-8") as summary_f, failures_path.open("w", encoding="utf-8") as fail_f:
        for idx, doc_path in enumerate(doc_paths, 1):
            try:
                with doc_path.open("r", encoding="utf-8") as f:
                    doc = json.load(f)

                schema = extract_schema(str(doc_path))
                module_fqn = _module_fqn_from_doc(doc)
                query = _build_probe_query(doc, schema, idx)

                if args.print_every and (idx == 1 or idx % args.print_every == 0):
                    print("=" * 100)
                    print(f"[{idx}/{total}] {query}")
                    print("DOC:", doc_path)

                yaml_text, meta = generate_parser_state_yaml(
                    query=query,
                    schema=schema,
                    module_fqn=module_fqn,
                    max_tokens=args.max_tokens,
                    debug=False,
                    return_metadata=True,
                )

                try:
                    parsed = yaml.safe_load(yaml_text)
                    yaml_ok = parsed is not None
                except Exception as exc:
                    yaml_ok = False
                    parse_error = str(exc)
                else:
                    parse_error = ""

                parser_log = meta.get("parser_log", []) or []
                decoder_log = meta.get("decoder_log", []) or []
                trigger_log = meta.get("trigger_log", []) or []

                if yaml_ok:
                    parsed_ok += 1
                if parser_log:
                    parser_docs += 1
                if decoder_log:
                    decoder_docs += 1
                if trigger_log:
                    trigger_docs += 1

                total_switches += int(meta.get("switch_count", 0) or 0)
                total_trigger_events += len(trigger_log)

                record = {
                    "index": idx,
                    "doc_path": str(doc_path),
                    "module_fqn": module_fqn,
                    "query": query,
                    "yaml_ok": yaml_ok,
                    "parse_error": parse_error,
                    "parser_log_len": len(parser_log),
                    "decoder_log_len": len(decoder_log),
                    "trigger_log_len": len(trigger_log),
                    "switch_count": int(meta.get("switch_count", 0) or 0),
                    "yaml": yaml_text,
                    "meta": meta,
                }
                summary_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if not yaml_ok:
                    failure = {
                        "index": idx,
                        "doc_path": str(doc_path),
                        "module_fqn": module_fqn,
                        "query": query,
                        "parse_error": parse_error,
                        "yaml": yaml_text,
                        "meta": meta,
                    }
                    fail_f.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    failures.append(failure)

            except Exception as exc:
                failure = {
                    "index": idx,
                    "doc_path": str(doc_path),
                    "error": repr(exc),
                }
                fail_f.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failures.append(failure)

                if args.print_every:
                    print("FAIL:", doc_path, repr(exc))

    print("\n" + "=" * 100)
    print("PHASE 3 CORPUS SUMMARY")
    print("=" * 100)
    print(f"Docs processed: {total}")
    print(f"YAML parse ok: {parsed_ok}/{total}")
    print(f"Docs with parser logs: {parser_docs}/{total}")
    print(f"Docs with decoder logs: {decoder_docs}/{total}")
    print(f"Docs with trigger logs: {trigger_docs}/{total}")
    print(f"Average switch count: {total_switches / total:.2f}")
    print(f"Average trigger events: {total_trigger_events / total:.2f}")
    print(f"Failures: {len(failures)}")
    print(f"Summary written to: {summary_path}")
    print(f"Failures written to: {failures_path}")


if __name__ == "__main__":
    main()