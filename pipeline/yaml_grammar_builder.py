from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _safe_rule_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", name).lower()


def _scalar_rule() -> str:
    return r"scalar ::= [A-Za-z0-9._/:+-]+"


def _yaml_atom(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9._/:+-]+", text):
        return text
    return json.dumps(text)


def _build_scalar_field(
    field: str,
    spec: Dict[str, Any],
    indent: str,
    path: str,
    value_hints: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], str]:
    lines: List[str] = []
    safe = _safe_rule_name(path)
    value_rule = f"{safe}value"
    line_rule = f"{safe}line"

    t = str(spec.get("type", "str")).lower()
    choices = spec.get("choices") or []

    hint = None
    if value_hints:
        hint = value_hints.get(path)
        if hint is None:
            hint = value_hints.get(field)

    if hint is not None and not isinstance(hint, (list, dict)):
        if t in {"bool", "boolean"}:
            lower = str(hint).lower()
            if lower in {"true", "false"}:
                lines.append(f'{value_rule} ::= "{lower}"')
                lines.append(f'{line_rule} ::= "{indent}{field}: " {value_rule} "\\n"')
                return lines, line_rule

        if t in {"int", "integer"} and re.fullmatch(r"[0-9]+", str(hint)):
            lines.append(f"{value_rule} ::= {hint}")
            lines.append(f'{line_rule} ::= "{indent}{field}: " {value_rule} "\\n"')
            return lines, line_rule

        lines.append(f"{value_rule} ::= {json.dumps(_yaml_atom(hint))}")
        lines.append(f'{line_rule} ::= "{indent}{field}: " {value_rule} "\\n"')
        return lines, line_rule

    if choices:
        enum_values = " | ".join(json.dumps(_yaml_atom(v)) for v in choices)
        lines.append(f"{value_rule} ::= {enum_values}")
        lines.append(f'{line_rule} ::= "{indent}{field}: " {value_rule} "\\n"')
        return lines, line_rule

    if t in {"bool", "boolean"}:
        lines.append(f'{value_rule} ::= "true" | "false"')
        lines.append(f'{line_rule} ::= "{indent}{field}: " {value_rule} "\\n"')
        return lines, line_rule

    if t in {"int", "integer"}:
        lines.append(f"{value_rule} ::= [0-9]+")
        lines.append(f'{line_rule} ::= "{indent}{field}: " {value_rule} "\\n"')
        return lines, line_rule

    lines.append(f"{value_rule} ::= scalar")
    lines.append(f'{line_rule} ::= "{indent}{field}: " {value_rule} "\\n"')
    return lines, line_rule


def _build_list_field(
    field: str,
    indent: str,
    path: str,
    value_hints: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], str]:
    lines: List[str] = []
    safe = _safe_rule_name(path)
    line_rule = f"{safe}line"
    body_rule = f"{safe}body"
    child_indent = indent + "    "

    hint = None
    if value_hints:
        hint = value_hints.get(path)
        if hint is None:
            hint = value_hints.get(field)

    if isinstance(hint, list) and hint:
        item_rules: List[str] = []
        for idx, item in enumerate(hint):
            item_rule = f"{safe}item{idx}"
            item_rules.append(item_rule)
            literal = f"{child_indent}- {_yaml_atom(item)}\n"
            lines.append(f"{item_rule} ::= {json.dumps(literal)}")
        lines.append(f"{body_rule} ::= " + " ".join(item_rules))
        lines.append(f'{line_rule} ::= "{indent}{field}:\\n" {body_rule}')
        return lines, line_rule

    item_rule = f"{safe}item"
    literal = f"{child_indent}- "
    lines.append(f'{item_rule} ::= {json.dumps(literal)} scalar "\\n"')
    lines.append(
        f"{body_rule} ::= {item_rule} | {item_rule} {item_rule} | {item_rule} {item_rule} {item_rule}"
    )
    lines.append(f'{line_rule} ::= "{indent}{field}:\\n" {body_rule}')
    return lines, line_rule


def _build_dict_field(
    field: str,
    spec: Dict[str, Any],
    indent: str,
    path: str,
    value_hints: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], str]:
    lines: List[str] = []
    safe = _safe_rule_name(path)
    line_rule = f"{safe}line"
    body_rule = f"{safe}body"
    child_indent = indent + "    "
    suboptions = spec.get("suboptions") or {}

    hint = None
    if value_hints:
        hint = value_hints.get(path)
        if hint is None:
            hint = value_hints.get(field)

    if isinstance(hint, dict) and hint:
        item_rules: List[str] = []
        for idx, (k, v) in enumerate(hint.items()):
            item_rule = f"{safe}{_safe_rule_name(str(k))}item{idx}"
            item_rules.append(item_rule)
            literal = f"{child_indent}{k}: {_yaml_atom(v)}\n"
            lines.append(f"{item_rule} ::= {json.dumps(literal)}")
        lines.append(f"{body_rule} ::= " + " ".join(item_rules))
        lines.append(f'{line_rule} ::= "{indent}{field}:\\n" {body_rule}')
        return lines, line_rule

    if suboptions:
        child_rules: List[str] = []
        for child_name, child_spec in suboptions.items():
            child_path = f"{path}__{child_name}"
            child_lines, child_rule = _build_field_rule(
                child_name,
                child_spec,
                indent=child_indent,
                path=child_path,
                value_hints=value_hints,
            )
            lines.extend(child_lines)
            child_rules.append(child_rule)

        lines.append(f"{body_rule} ::= " + " ".join(child_rules))
        lines.append(f'{line_rule} ::= "{indent}{field}:\\n" {body_rule}')
        return lines, line_rule

    pair_rule = f"{safe}pair"
    literal = f"{child_indent}"
    lines.append(f'{pair_rule} ::= {json.dumps(literal)} scalar ": " scalar "\\n"')
    lines.append(
        f"{body_rule} ::= {pair_rule} | {pair_rule} {pair_rule} | {pair_rule} {pair_rule} {pair_rule}"
    )
    lines.append(f'{line_rule} ::= "{indent}{field}:\\n" {body_rule}')
    return lines, line_rule


def _build_field_rule(
    field: str,
    spec: Dict[str, Any],
    indent: str,
    path: str,
    value_hints: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], str]:
    hint = None
    if value_hints:
        hint = value_hints.get(path)
        if hint is None:
            hint = value_hints.get(field)

    if isinstance(hint, list):
        return _build_list_field(field, indent, path, value_hints=value_hints)
    if isinstance(hint, dict):
        return _build_dict_field(field, spec, indent, path, value_hints=value_hints)

    t = str(spec.get("type", "str")).lower()
    if t in {"list", "array"}:
        return _build_list_field(field, indent, path, value_hints=value_hints)
    if t in {"dict", "mapping", "object"} or spec.get("suboptions"):
        return _build_dict_field(field, spec, indent, path, value_hints=value_hints)
    return _build_scalar_field(field, spec, indent, path, value_hints=value_hints)


def build_yaml_grammar(
    module_fqn: str,
    schema: dict,
    include_fields: Optional[List[str]] = None,
    value_hints: Optional[Dict[str, Any]] = None,
    include_header: bool = True,
) -> str:
    types = schema.get("types", {}) or {}
    choices = schema.get("choices", {}) or {}
    suboptions = schema.get("suboptions", {}) or {}

    if include_fields is None:
        fields = list(schema.get("required", []))
    else:
        fields = list(include_fields)

    lines: List[str] = []
    lines.append(_scalar_rule())
    lines.append("")
    lines.append("string ::= scalar")
    lines.append("")

    field_rule_names: List[str] = []

    for field in fields:
        spec = {
            "type": types.get(field, "str"),
            "choices": choices.get(field, []),
            "suboptions": suboptions.get(field, {}),
        }

        field_lines, field_rule = _build_field_rule(
            field=field,
            spec=spec,
            indent="    ",
            path=field,
            value_hints=value_hints,
        )
        lines.extend(field_lines)
        lines.append("")
        field_rule_names.append(field_rule)

    if include_header:
        lines.append(f'header ::= "- name: Generated Task\\n  {module_fqn}:\\n"')
        if not field_rule_names:
            lines.append("root ::= header")
        else:
            lines.append("root ::= header " + " ".join(field_rule_names))
    else:
        if not field_rule_names:
            lines.append('root ::= ""')
        else:
            lines.append("root ::= " + " ".join(field_rule_names))

    return "\n".join(lines)