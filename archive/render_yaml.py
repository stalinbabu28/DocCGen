from __future__ import annotations

import json
import re
from typing import Any, Dict


_SIMPLE_SCALAR_RE = re.compile(r"^[A-Za-z0-9_./:@+\-]+$")


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        if _SIMPLE_SCALAR_RE.fullmatch(value):
            return value
        return json.dumps(value)

    return json.dumps(str(value))


def _should_skip(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "?", "??", "???", "none", "null", "n/a", "na", "unknown"}:
            return True
        if "..." in s:
            return True
        if re.fullmatch(r"[\?\.\-_ ]+", s):
            return True
    return False


def _fmt_value(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = ", ".join(f"{k}: {_fmt_value(v)}" for k, v in value.items() if not _should_skip(v))
        return "{ " + inner + " }"

    if isinstance(value, list):
        items = [v for v in value if not _should_skip(v)]
        if not items:
            return "[]"
        return "[ " + ", ".join(_fmt_value(v) for v in items) + " ]"

    return _fmt_scalar(value)


def render_yaml(module_name: str, params: Dict[str, Any], task_name: str = "Generated Task") -> str:
    lines = [
        f"- name: {task_name}",
        f"  {module_name}:",
    ]

    for key, value in params.items():
        if _should_skip(value):
            continue
        lines.append(f"    {key}: {_fmt_value(value)}")

    return "\n".join(lines)