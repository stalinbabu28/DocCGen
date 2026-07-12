from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

NAME_RE = r"[A-Za-z0-9._/-]+(?:\.[A-Za-z0-9._/-]+)*"
IPV4_RE = r"(?:\d{1,3}\.){3}\d{1,3}"

BOOL_TRUE = {"true", "yes", "1", "on", "enabled"}
BOOL_FALSE = {"false", "no", "0", "off", "disabled"}

STOPWORDS = {
    "and", "or", "the", "a", "an", "of", "to", "for", "with", "from", "in",
    "on", "as", "via", "using", "then", "by", "into", "for", "this", "that",
    "these", "those", "is", "are", "be", "become", "becomes"
}


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_type(raw: Any) -> str:
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


def unique(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def is_info_module(module_fqn: Optional[str]) -> bool:
    return bool(module_fqn and (module_fqn.endswith("_info") or module_fqn.endswith("_facts")))


def alias_list(schema: Dict[str, Any], field: str) -> List[str]:
    aliases = schema.get("aliases", {}) or {}
    raw = aliases.get(field, [])
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def field_phrases(schema: Dict[str, Any], field: str) -> List[str]:
    phrases = [
        field.lower(),
        field.replace("_", " ").lower(),
        field.replace("_", "").lower(),
    ]
    phrases.extend(a.lower() for a in alias_list(schema, field))
    return unique(p for p in phrases if p.strip())


def phrase_in_query(query: str, phrase: str) -> bool:
    q = collapse_ws(query)
    p = collapse_ws(phrase)
    if not p:
        return False
    # Note: a plain \b / (?<!\w) boundary treats "-" as a non-word character,
    # so it would match "src" inside "demo-src". Excluding "-" (and other
    # NAME_RE characters) from the boundary prevents matching a phrase that is
    # actually just a substring of a hyphenated/dotted token in the query.
    return re.search(rf"(?<![A-Za-z0-9_.-]){re.escape(p)}(?![A-Za-z0-9_.-])", q) is not None


def first_match(text: str, patterns: Sequence[str]) -> Optional[str]:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def token_list(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9._/-]+", text)
    return [t for t in tokens if t.lower() not in STOPWORDS]


def clean_scalar(value: Any) -> str:
    s = str(value).strip()
    s = s.strip(" \t\r\n'\"")
    s = s.rstrip(".,;:!?")
    return s

def extract_after_phrase_segment(query: str, phrases: Sequence[str]) -> Optional[str]:
    """
    Return the text that appears after the earliest matching phrase in `phrases`.

    Example:
        query  = "create virtual network called prod-vnet in resource group prod-rg"
        phrases = ["virtual network"]
        -> "called prod-vnet in resource group prod-rg"

    The returned fragment is cleaned a bit so downstream extractors can more
    reliably parse values from it.
    """
    if not query or not phrases:
        return None

    q = collapse_ws(query)

    best_idx = None
    best_phrase = None

    for phrase in phrases:
        p = collapse_ws(phrase)
        if not p:
            continue
        idx = q.find(p)
        if idx == -1:
            continue
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_phrase = p

    if best_idx is None or best_phrase is None:
        return None

    segment = q[best_idx + len(best_phrase):].strip()

    # Remove leading separators / cue words.
    segment = re.sub(
        r"^(?:[:=,\-]\s*|\b(?:called|named|with|of|to|from|in|for|as|using|via)\b\s*)+",
        "",
        segment,
        flags=re.IGNORECASE,
    )

    # Stop at common follow-on clauses.
    segment = re.split(
        r"\b(?:and\s+append\s+tags|and\s+resource\s+group|and\s+location|and\s+region|and\s+dns\s+servers?|and\s+tags?|then|using|via)\b",
        segment,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    # Stop at sentence endings.
    segment = re.split(r"[.;]\s*", segment, maxsplit=1)[0]

    return segment.strip(" ,")

def normalize_bool(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"true", "yes", "1", "on", "enabled"}:
        return "true"
    if v in {"false", "no", "0", "off", "disabled"}:
        return "false"
    return str(value).strip()