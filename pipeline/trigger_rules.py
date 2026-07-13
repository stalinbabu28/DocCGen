from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional

from pipeline.schema_text_utils import (
    collapse_ws,
    field_phrases,
    is_info_module,
    normalize_type,
    phrase_in_query,
    unique,
)

DELETE_CUES = (" delete ", " remove ", " absent ")
BOOL_CUES = (" true ", " false ", " yes ", " no ", " 1 ", " 0 ", " on ", " off ", " enabled ", " disabled ")

SPECIAL_LISTISH_WORDS = {
    "list", "items", "item", "servers", "addresses", "names", "ids", "resources",
    "entries", "subnets", "rules", "neighbors", "trail_names", "networks", "ranges"
}

SPECIAL_DICTISH_WORDS = {
    "tags", "tag", "labels", "label", "annotations", "metadata",
    "options", "settings", "parameters", "args", "arguments", "attributes", "properties"
}

GENERIC_NAME_FIELDS = {
    "name", "resource_group", "location", "region", "subnet_name",
    "virtual_network_name", "virtual_network", "path", "src", "dest", "url", "line"
}


def _has_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def _explicit_resource_group_trigger(query: str) -> bool:
    return phrase_in_query(query, "resource group")


def _explicit_state_trigger(query: str) -> bool:
    q = collapse_ws(query)
    return any(cue in f" {q} " for cue in DELETE_CUES)


def _explicit_name_trigger(query: str) -> bool:
    q = collapse_ws(query)
    return phrase_in_query(q, "named") or phrase_in_query(q, "called") or phrase_in_query(q, "name ")


def _explicit_tags_trigger(query: str) -> bool:
    q = collapse_ws(query)
    return phrase_in_query(q, "tags") or phrase_in_query(q, "tag") or phrase_in_query(q, "labels")


def _explicit_location_trigger(query: str) -> bool:
    q = collapse_ws(query)
    return phrase_in_query(q, "location") or phrase_in_query(q, "region")


@dataclass
class GrammarState:
    active_fields: List[str]
    triggered: List[str] = dataclass_field(default_factory=list)


def project_schema(schema: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    wanted = set(fields)
    projected = deepcopy(schema)

    for key in ("required", "optional"):
        if key in projected:
            projected[key] = [f for f in projected.get(key, []) if f in wanted]

    for key in ("choices", "types", "descriptions", "defaults", "aliases", "elements", "dependencies"):
        if key not in projected or not isinstance(projected[key], dict):
            continue
        projected[key] = {f: deepcopy(v) for f, v in projected[key].items() if f in wanted}

    if "suboptions" in projected and isinstance(projected["suboptions"], dict):
        projected["suboptions"] = {f: deepcopy(v) for f, v in projected["suboptions"].items() if f in wanted}

    return projected

def _explicit_src_trigger(query: str) -> bool:
    q = collapse_ws(query)
    return phrase_in_query(q, "from") or phrase_in_query(q, "source") or phrase_in_query(q, "using")


def _explicit_dest_trigger(query: str) -> bool:
    q = collapse_ws(query)
    return (
        phrase_in_query(q, "to")
        or phrase_in_query(q, "into")
        or phrase_in_query(q, "dest")
        or phrase_in_query(q, "destination")
        or phrase_in_query(q, "target")
    )


def infer_active_fields(
    query: str,
    schema: Dict[str, Any],
    module_fqn: Optional[str] = None,
) -> GrammarState:
    """
    Conservative semantic trigger pass.
    """
    q = f" {collapse_ws(query)} "
    info_module = is_info_module(module_fqn)

    required = list(schema.get("required", []))
    optional = list(schema.get("optional", []))
    all_fields = unique(required + [f for f in optional if f not in required])

    active: List[str] = []
    triggered: List[str] = []

    def add(field: str, reason: str) -> None:
        if field in all_fields and field not in active:
            active.append(field)
            triggered.append(f"{field}:{reason}")

    if _has_any(q, DELETE_CUES) and "state" in all_fields:
        add("state", "delete-trigger")

    for field in all_fields:
        if field == "state":
            continue

        t = normalize_type(schema.get("types", {}).get(field, "str"))
        phrases = field_phrases(schema, field)

        if field in {"src", "source"} and _explicit_src_trigger(query):
            add(field, "src-directional-cue")
            continue

        if field in {"dest", "destination", "target"} and _explicit_dest_trigger(query):
            add(field, "dest-directional-cue")
            continue

        # Boolean fields: do not activate from bare action verbs like "create".
        if t == "bool":
            if field in {"append_tags", "backup", "force", "enabled", "disabled", "validate", "check", "public", "private"}:
                if any(phrase_in_query(q, p) for p in phrases) and any(cue in q for cue in BOOL_CUES):
                    add(field, "bool-cue")
            continue

        if any(phrase_in_query(q, p) for p in phrases):
            if field in GENERIC_NAME_FIELDS:
                if field == "name":
                    add(field, "name-cue")
                else:
                    add(field, "field-cue")
                continue

            if t in {"list", "dict"}:
                add(field, f"{t}-cue")
                continue

            add(field, "field-cue")
            continue

        if field == "append_tags" and phrase_in_query(query, "append tags"):
            add(field, "append-tags-cue")
            continue

        if field == "tags" and _explicit_tags_trigger(query):
            add(field, "tags-cue")
            continue

        if field == "resource_group" and _explicit_resource_group_trigger(query):
            add(field, "resource-group-cue")
            continue

        if field == "name" and _explicit_name_trigger(query):
            add(field, "name-cue")
            continue

        if field in {"location", "region"} and _explicit_location_trigger(query):
            add(field, "location-cue")
            continue

        if info_module:
            continue

        if t == "list" and any(word in q for word in SPECIAL_LISTISH_WORDS):
            if any(phrase_in_query(q, p) for p in phrases):
                add(field, "list-cue")
        elif t == "dict" and any(word in q for word in SPECIAL_DICTISH_WORDS):
            if any(phrase_in_query(q, p) for p in phrases):
                add(field, "dict-cue")

    if not active:
        active = [f for f in required if f in all_fields]
        for f in active:
            triggered.append(f"{f}:required-fallback")

    return GrammarState(active_fields=active, triggered=triggered)