from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def strip_wrappers(text: str) -> str:
    text = text.strip()
    for token in [
        "```json",
        "```",
        "<|end|>",
        "<|start|>",
        "<|assistant|>",
        "<|user|>",
    ]:
        text = text.replace(token, "")
    return text.strip()


def _balanced_json_candidates(text: str):
    """
    Yield balanced {...} substrings from the text.
    We try every '{' position and return any balanced region we can find.
    """
    starts = [i for i, ch in enumerate(text) if ch == "{"]

    for start in starts:
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Return the last valid JSON object found in noisy model output.
    """
    text = strip_wrappers(text)

    candidates = list(_balanced_json_candidates(text))
    if not candidates:
        return None

    # Prefer the last valid JSON object, since the model often explains first
    # and then places the intended JSON at the end.
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def _is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "?", "??", "???", "none", "null", "n/a", "na", "unknown"}:
            return True
        if re.fullmatch(r"[\?\.\-_, ]+", s):
            return True
        if "..." in s:
            return True

    return False


def normalize_params(
    parsed: Optional[Dict[str, Any]],
    doc: dict,
    query: str = "",
    drop_unknown: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Normalize aliases to canonical schema names.
    Optionally drop unknown keys.
    Also removes obvious placeholder values.
    """
    if parsed is None:
        return None

    options = doc.get("options", {}) or {}
    canonical_keys = set(options.keys())

    alias_to_canonical = {}
    for canonical, meta in options.items():
        if not isinstance(meta, dict):
            continue
        for alias in meta.get("aliases", []) or []:
            alias_to_canonical[alias] = canonical

    normalized: Dict[str, Any] = {}

    for key, value in parsed.items():
        canonical = alias_to_canonical.get(key, key)

        if drop_unknown and canonical not in canonical_keys:
            continue

        if _is_placeholder_value(value):
            continue

        normalized[canonical] = value

    q = query.lower()
    if ("delete" in q or "remove" in q or "absent" in q) and "state" in canonical_keys:
        normalized["state"] = "absent"

    return normalized