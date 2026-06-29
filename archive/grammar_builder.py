from __future__ import annotations

import re
from typing import List, Optional


def _safe_rule_name(name: str) -> str:
    """
    Make a llama.cpp-safe rule name.
    Only letters and digits; lowercased.
    """
    return re.sub(r"[^A-Za-z0-9]+", "", name).lower()


def _json_quoted_literal(value: str) -> str:
    """
    Produce a grammar terminal for a JSON string literal.
    Example:
        "\"present\""
    """
    return f'"\\"{value}\\""'


def build_grammar(schema: dict, include_fields: Optional[List[str]] = None) -> str:
    """
    Build a llama.cpp GBNF grammar for the selected fields.

    Supported field kinds:
    - string (default)
    - enum via schema["choices"][field]
    - bool
    - int
    - list      -> one string item, e.g. ["10.0.0.0/16"]
    - dict      -> one key/value pair, e.g. {"env":"prod"}

    This is intentionally conservative and stable for llama-cpp-python 0.3.28.
    """
    types = schema.get("types", {}) or {}
    choices = schema.get("choices", {}) or {}

    if include_fields is None:
        fields = list(schema.get("required", []))
    else:
        fields = list(include_fields)

    lines: List[str] = []

    # Shared string rule. This version has already been working for you.
    lines.append(r'string ::= "\"" [a-zA-Z0-9._/ -]+ "\""')
    lines.append("")

    pair_rules: List[str] = []

    for field in fields:
        safe = _safe_rule_name(field)
        pair_rule = f"{safe}pair"
        pair_rules.append(pair_rule)

        field_type = str(types.get(field, "str")).lower()

        # Enum / choices
        if field in choices:
            value_rule = f"{safe}value"
            enum_values = [_json_quoted_literal(str(v)) for v in choices[field]]
            lines.append(f'{value_rule} ::= {" | ".join(enum_values)}')
            lines.append(f'{pair_rule} ::= "\\"{field}\\"" ":" {value_rule}')
            lines.append("")
            continue

        # Boolean
        if field_type in {"bool", "boolean"}:
            value_rule = f"{safe}value"
            lines.append(f'{value_rule} ::= "true" | "false"')
            lines.append(f'{pair_rule} ::= "\\"{field}\\"" ":" {value_rule}')
            lines.append("")
            continue

        # Integer
        if field_type in {"int", "integer"}:
            value_rule = f"{safe}value"
            lines.append(f"{value_rule} ::= [0-9]+")
            lines.append(f'{pair_rule} ::= "\\"{field}\\"" ":" {value_rule}')
            lines.append("")
            continue

        # List: one string item
        if field_type in {"list", "array"}:
            value_rule = f"{safe}value"
            lines.append(f'{value_rule} ::= "[" string "]"')
            lines.append(f'{pair_rule} ::= "\\"{field}\\"" ":" {value_rule}')
            lines.append("")
            continue

        # Dict: one key/value pair
        if field_type in {"dict", "mapping", "object"}:
            value_rule = f"{safe}value"
            lines.append(f'{value_rule} ::= "{{" string ":" string "}}"')
            lines.append(f'{pair_rule} ::= "\\"{field}\\"" ":" {value_rule}')
            lines.append("")
            continue

        # Default string
        lines.append(f'{pair_rule} ::= "\\"{field}\\"" ":" string')
        lines.append("")

    if not pair_rules:
        lines.append('root ::= "{}"')
    else:
        root_body = ' "," '.join(pair_rules)
        lines.append('root ::= "{" ' + root_body + ' "}"')

    return "\n".join(lines) 