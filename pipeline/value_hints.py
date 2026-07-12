from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pipeline.schema_text_utils import (
    IPV4_RE,
    NAME_RE,
    clean_scalar,
    collapse_ws,
    extract_after_phrase_segment,
    field_phrases,
    first_match,
    normalize_bool,
    normalize_type,
    phrase_in_query,
    token_list,
    unique,
)

BAD_SCALAR_VALUES = {
    "and", "or", "the", "a", "an", "of", "to", "for", "with", "from", "in",
    "on", "as", "via", "using", "true", "false", "yes", "no", "delete",
    "remove", "show", "list", "create", "details", "information", "called",
    "named", "ensure", "set", "make"
}

DEFAULT_PLACEHOLDERS = {
    "name": "demo-resource",
    "path": "demo-path",
    "src": "demo-src",
    "dest": "demo-dest",
    "url": "demo-url",
    "line": "demo-line",
    "bucket": "demo-bucket",
    "resource_group": "prod-rg",
    "region": "us-east-1",
    "location": "us-east-1",
    "vpc": "prod-vpc",
    "subnet": "subnet-a",
    "subnet_name": "subnet-a",
    "virtual_network_name": "prod-vnet",
    "virtual_network": "prod-vnet",
    "security_group": "web-sg",
    "key_pair": "demo-key",
    "instance_id": "i-1234567890abcdef0",
    "database": "inventory-db",
    "key_vault": "company-vault",
    "state": "present",
}

BOOL_FIELDS = {
    "backup", "force", "enabled", "disabled", "validate", "check",
    "public", "private", "append_tags", "purge", "overwrite", "recursive"
}


def _looks_bad_scalar(value: str) -> bool:
    v = clean_scalar(value).lower()
    if not v:
        return True
    if v in BAD_SCALAR_VALUES:
        return True
    if len(v) == 1 and not v.isdigit():
        return True
    if re.fullmatch(r"[.]+", v):
        return True
    return False


def _choice_value(query: str, choices: List[Any]) -> Optional[Any]:
    q = collapse_ws(query)
    for choice in choices:
        if choice is None:
            continue
        c = clean_scalar(choice)
        if c and phrase_in_query(q, c.lower()):
            return choice
    return None


def _extract_bool(query: str, phrases: List[str]) -> Optional[bool]:
    q = collapse_ws(query)
    for phrase in phrases:
        tail = extract_after_phrase_segment(query, [phrase])
        if tail:
            m = re.search(r"\b(true|false|yes|no|1|0|on|off|enabled|disabled)\b", tail, flags=re.IGNORECASE)
            if m:
                return normalize_bool(m.group(1)) == "true"

    for token in ("true", "false", "yes", "no", "enabled", "disabled", "on", "off"):
        if f" {token} " in f" {q} ":
            return normalize_bool(token) == "true"

    return None


# A plain \b boundary treats "-" (and ".", "/") as non-word characters, so
# `\bsrc\b` will happily match the "src" inside "demo-src" — then the pattern
# goes on to capture whatever word follows ("into", etc.) as the field value.
# PHRASE_BOUNDARY excludes those characters too, so a phrase only matches when
# it's a standalone token in the query, not a substring of a hyphenated /
# dotted placeholder value.
PHRASE_BOUNDARY_PRE = r"(?<![A-Za-z0-9_.-])"
PHRASE_BOUNDARY_POST = r"(?![A-Za-z0-9_.-])"


def _phrase_pattern(phrase: str) -> str:
    return f"{PHRASE_BOUNDARY_PRE}{re.escape(phrase)}{PHRASE_BOUNDARY_POST}"


def _extract_int(query: str, phrases: List[str]) -> Optional[int]:
    patterns: List[str] = []
    for phrase in phrases:
        p = _phrase_pattern(phrase)
        patterns.extend([
            rf"{p}\s+(?:called|named|with|of|to|from|in|for|as|=|:)\s+([0-9]+)",
            rf"{p}\s+([0-9]+)",
        ])
    m = first_match(query, patterns)
    if m and re.fullmatch(r"[0-9]+", m):
        return int(m)
    return None


def _extract_scalar(query: str, phrases: List[str]) -> Optional[str]:
    patterns: List[str] = []
    for phrase in phrases:
        p = _phrase_pattern(phrase)
        patterns.extend([
            rf"{p}\s+(?:called|named|with|of|to|from|in|for|as|at|=|:)\s+({NAME_RE})",
            rf"{p}\s+({NAME_RE})",
        ])

    m = first_match(query, patterns)
    if m:
        m = clean_scalar(m)
        if not _looks_bad_scalar(m):
            return m
    return None


def _extract_list(query: str, phrases: List[str]) -> Optional[List[str]]:
    for phrase in phrases:
        tail = extract_after_phrase_segment(query, [phrase])
        if tail:
            ips = unique(re.findall(IPV4_RE, tail))
            if ips:
                return [clean_scalar(x) for x in ips]

            toks = [clean_scalar(t) for t in token_list(tail) if not _looks_bad_scalar(t)]
            if len(toks) >= 2:
                return toks[:16]

    ips = unique(re.findall(IPV4_RE, query))
    if len(ips) >= 2:
        return [clean_scalar(x) for x in ips]

    return None


def _extract_dict(query: str, phrases: List[str]) -> Optional[Dict[str, Any]]:
    for phrase in phrases:
        tail = extract_after_phrase_segment(query, [phrase])
        if not tail:
            continue

        kv_pairs = re.findall(r"([A-Za-z0-9._/-]+)\s*[:=]\s*([A-Za-z0-9._/-]+)", tail)
        if kv_pairs:
            return {clean_scalar(k): clean_scalar(v) for k, v in kv_pairs}

        toks = [clean_scalar(t) for t in token_list(tail) if not _looks_bad_scalar(t)]
        if len(toks) >= 4:
            out: Dict[str, Any] = {}
            i = 0
            while i + 1 < len(toks):
                out[toks[i]] = toks[i + 1]
                i += 2
            if out:
                return out

    return None


def _placeholder_for_field(field: str) -> Optional[Any]:
    if field in DEFAULT_PLACEHOLDERS:
        return DEFAULT_PLACEHOLDERS[field]

    lower = field.lower()
    for key, value in DEFAULT_PLACEHOLDERS.items():
        if key in lower:
            return value

    if "path" in lower:
        return "demo-path"
    if "src" in lower:
        return "demo-src"
    if "dest" in lower:
        return "demo-dest"
    if "url" in lower:
        return "demo-url"
    if "line" in lower:
        return "demo-line"
    if "name" in lower:
        return "demo-resource"
    if "group" in lower:
        return "prod-rg"
    if "region" in lower:
        return "us-east-1"
    if "subnet" in lower:
        return "subnet-a"
    if "vpc" in lower:
        return "prod-vpc"
    if "bucket" in lower:
        return "demo-bucket"
    if "state" in lower:
        return "present"
    return None


def infer_value_hints(
    query: str,
    schema: Dict[str, Any],
    module_fqn: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Schema-driven value extraction with punctuation cleanup.
    """
    hints: Dict[str, Any] = {}
    q = query.strip()
    ql = f" {collapse_ws(q)} "
    types = schema.get("types", {}) or {}
    choices = schema.get("choices", {}) or {}

    if "state" in types:
        if " delete " in ql or " remove " in ql:
            hints["state"] = "absent"
        elif " present " in ql or " ensure " in ql or " create " in ql or " update " in ql:
            hints["state"] = "present"

    for field, raw_type in types.items():
        if field in hints:
            continue

        t = normalize_type(raw_type)
        phrases = field_phrases(schema, field)
        choice_list = choices.get(field, []) or []

        choice = _choice_value(q, choice_list)
        if choice is not None:
            hints[field] = choice
            continue

        if t == "bool":
            v = _extract_bool(q, phrases)
            if v is not None:
                hints[field] = v
                continue

        if t == "int":
            v = _extract_int(q, phrases)
            if v is not None:
                hints[field] = v
                continue

        if t == "list":
            v = _extract_list(q, phrases)
            if v is not None:
                hints[field] = v
                continue

        if t == "dict":
            v = _extract_dict(q, phrases)
            if v is not None:
                hints[field] = v
                continue

        v = _extract_scalar(q, phrases)
        if v is not None:
            hints[field] = v
            continue

        placeholder = _placeholder_for_field(field)
        if placeholder is not None:
            hints[field] = placeholder

    return hints