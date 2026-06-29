from __future__ import annotations

from typing import List


INFO_CUES = ("show ", "details", "information", "info", "get ")
NAME_CUES = (" named ", " called ")
DELETE_CUES = ("delete", "remove", " absent ")


def _has_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def select_fields(query: str, schema: dict) -> List[str]:
    q = f" {query.lower()} "

    required = list(schema.get("required", []))
    optional = list(schema.get("optional", []))
    all_fields = required + [f for f in optional if f not in required]

    selected: List[str] = []

    def add(field: str) -> None:
        if field in all_fields and field not in selected:
            selected.append(field)

    # Delete/remove: include state and name if available.
    if _has_any(q, DELETE_CUES):
        if "state" in all_fields:
            add("state")
        if "name" in all_fields:
            add("name")

    # Direct mentions by field name or normalized name.
    for field in all_fields:
        normalized = field.replace("_", " ")
        if field in q or normalized in q:
            add(field)

    # Common field-specific heuristics.
    if "name" in all_fields and (
        _has_any(q, NAME_CUES) or _has_any(q, INFO_CUES) or _has_any(q, DELETE_CUES)
    ):
        add("name")

    if "resource_group" in all_fields and "resource group" in q:
        add("resource_group")

    if "virtual_network_name" in all_fields and "virtual network" in q:
        add("virtual_network_name")

    if "subnet_name" in all_fields and "subnet" in q:
        add("subnet_name")

    return selected