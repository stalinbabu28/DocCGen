from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional

NAME_RE = r"[A-Za-z0-9._/-]+(?:\.[A-Za-z0-9._/-]+)*"

DELETE_CUES = (" delete ", " remove ", " absent ")
INFO_CUES = (" show ", " details ", " information ", " info ", " get ", " list ", " display ", " describe ")

SENSITIVE_FIELDS = {
    "name",
    "resource_group",
    "state",
    "tags",
    "append_tags",
    "dns_servers",
    "virtual_network_name",
    "virtual_network",
    "subnet_name",
    "location",
}


def _has_any(text: str, phrases) -> bool:
    return any(p in text for p in phrases)


def _is_info_module(module_fqn: Optional[str]) -> bool:
    return bool(module_fqn and module_fqn.endswith("_info"))


def _has_tags_kv(query: str) -> bool:
    if re.search(r"\b(?:with\s+)?tags?\b", query, flags=re.IGNORECASE) is None:
        return False

    if re.search(r"\bappend\s+tags\b", query, flags=re.IGNORECASE):
        # "append tags false" should not activate tags
        if not re.search(r"\b(?:with\s+)?tags?\b.*\b[A-Za-z0-9._/-]+\s+[A-Za-z0-9._/-]+\b", query, flags=re.IGNORECASE | re.DOTALL):
            return False

    m = re.search(r"\b(?:with\s+)?tags?\b(.*)$", query, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return False

    segment = m.group(1)
    if re.search(r"\bappend\s+tags\b", segment, flags=re.IGNORECASE):
        segment = re.split(r"\band\s+append\s+tags\b", segment, maxsplit=1, flags=re.IGNORECASE)[0]

    if re.search(r"([A-Za-z0-9._/-]+)\s*[:=]\s*([A-Za-z0-9._/-]+)", segment):
        return True

    tokens = re.findall(r"[A-Za-z0-9._/-]+", segment)
    tokens = [t for t in tokens if t.lower() not in {"and", "with", "tags", "tag", "append", "true", "false", "yes", "no"}]
    return len(tokens) >= 4


def _generic_field_trigger(field: str, query: str) -> bool:
    if field in SENSITIVE_FIELDS:
        return False

    phrase = field.replace("_", " ")
    patterns = [
        rf"\b{re.escape(phrase)}\s+(?:called|named|with|of|to|from|in|for)\s+({NAME_RE})",
        rf"\b{re.escape(phrase)}\s+({NAME_RE})",
    ]
    return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)


def _explicit_name_trigger(query: str) -> bool:
    patterns = [
        rf"\bdelete\s+subnet\s+({NAME_RE})",
        rf"\b(?:create|delete|show|get|list|update)\s+a?\s*subnet\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\b(?:virtual network gateway|virtual network|dns zone|public ip address|route table|storage account|managed disk|load balancer|application gateway|web app|network interface|virtual machine scale set|virtual machine|aks cluster|container registry|key vault|sql database|sql server)\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\b(?:show details of|get information about|list|display|describe)\s+(?:a|an|the)?\s*(?:virtual network gateway|virtual network|dns zone|public ip address|route table|storage account|managed disk|load balancer|application gateway|web app|network interface|virtual machine scale set|virtual machine|aks cluster|container registry|key vault|sql database|sql server)\s+({NAME_RE})",
        rf"\bnamed\s+({NAME_RE})",
        rf"\bcalled\s+({NAME_RE})",
    ]
    return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)


def _explicit_resource_group_trigger(query: str) -> bool:
    return re.search(r"\bresource group\s+([A-Za-z0-9._/-]+)", query, flags=re.IGNORECASE) is not None


def _explicit_subnet_name_trigger(query: str) -> bool:
    patterns = [
        rf"\bdelete\s+subnet\s+({NAME_RE})",
        rf"\b(?:create|delete|show|get|list|update)\s+a?\s*subnet\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bsubnet\s+(?:called\s+|named\s+)?({NAME_RE})",
    ]
    return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)


def _explicit_virtual_network_name_trigger(query: str) -> bool:
    patterns = [
        rf"\b(?:in|on|with|to|attached to)\s+virtual network\s+({NAME_RE})",
        rf"\bvirtual network\s+(?:called\s+|named\s+)?({NAME_RE})",
    ]
    # Exclude phrases like "virtual network gateway" from being treated as a VNet name.
    for p in patterns:
        m = re.search(p, query, flags=re.IGNORECASE)
        if m and m.group(1).lower() != "gateway":
            return True
    return False


def _explicit_virtual_network_value_trigger(query: str) -> bool:
    patterns = [
        rf"\b(?:in|on|with|to|attached to)\s+virtual network\s+({NAME_RE})",
    ]
    return any(re.search(p, query, flags=re.IGNORECASE) for p in patterns)


def _explicit_dns_trigger(query: str) -> bool:
    return "dns server" in query.lower() or "dns servers" in query.lower()


def _explicit_append_tags_trigger(query: str) -> bool:
    return re.search(r"\bappend\s+tags\s+(true|false|yes|no)\b", query, flags=re.IGNORECASE) is not None


def _explicit_location_trigger(query: str) -> bool:
    # Do not use bare "in" because that causes bad false positives.
    return re.search(
        r"\b(?:location|region)\s+([A-Za-z0-9._/-]+)\b",
        query,
        flags=re.IGNORECASE,
    ) is not None


@dataclass
class GrammarState:
    active_fields: List[str]
    triggered: List[str] = field(default_factory=list)


def infer_active_fields(query: str, schema: Dict[str, Any], module_fqn: Optional[str] = None) -> GrammarState:
    q = f" {query.lower()} "
    info_module = _is_info_module(module_fqn)

    required = list(schema.get("required", []))
    optional = list(schema.get("optional", []))
    all_fields = required + [f for f in optional if f not in required]

    active: List[str] = []
    triggered: List[str] = []

    def add(field: str, reason: str) -> None:
        if field in all_fields and field not in active:
            active.append(field)
            triggered.append(f"{field}:{reason}")

    # State only for delete/remove requests.
    if _has_any(q, DELETE_CUES) and "state" in all_fields:
        add("state", "delete-trigger")

    # Resource group only when explicit.
    if "resource_group" in all_fields and _explicit_resource_group_trigger(query):
        add("resource_group", "resource-group-cue")

    # Name only when explicit or clearly embedded in an info/list request.
    if "name" in all_fields and _explicit_name_trigger(query):
        add("name", "name-cue")

    if info_module:
        # Info modules should be conservative: only activate fields that are explicitly asked for.
        if "virtual_network_name" in all_fields and _explicit_virtual_network_name_trigger(query):
            add("virtual_network_name", "info-virtual-network")
        if "subnet_name" in all_fields and _explicit_subnet_name_trigger(query):
            add("subnet_name", "info-subnet")
        if "dns_servers" in all_fields and _explicit_dns_trigger(query):
            add("dns_servers", "info-dns")
        if "append_tags" in all_fields and _explicit_append_tags_trigger(query):
            add("append_tags", "info-append-tags")
        if "tags" in all_fields and _has_tags_kv(query):
            add("tags", "info-tags")
        return GrammarState(active_fields=active, triggered=triggered)

    # Non-info modules: explicit structured fields only.
    if "append_tags" in all_fields and _explicit_append_tags_trigger(query):
        add("append_tags", "append-tags-cue")

    if "tags" in all_fields and _has_tags_kv(query):
        add("tags", "tags-cue")

    if "dns_servers" in all_fields and _explicit_dns_trigger(query):
        add("dns_servers", "dns-cue")

    if "subnet_name" in all_fields and _explicit_subnet_name_trigger(query):
        add("subnet_name", "subnet-cue")

    if "virtual_network_name" in all_fields and _explicit_virtual_network_name_trigger(query):
        add("virtual_network_name", "virtual-network-name-cue")

    if "virtual_network" in all_fields and _explicit_virtual_network_value_trigger(query):
        add("virtual_network", "virtual-network-cue")

    if "location" in all_fields and _explicit_location_trigger(query):
        add("location", "location-cue")

    # Generic fallback only for non-sensitive fields.
    for field in all_fields:
        if field in active:
            continue
        if _generic_field_trigger(field, query):
            add(field, "generic-field-cue")

    return GrammarState(active_fields=active, triggered=triggered)


def project_schema(schema: Dict[str, Any], active_fields: List[str]) -> Dict[str, Any]:
    active = set(active_fields)
    out = deepcopy(schema)

    out["required"] = [f for f in schema.get("required", []) if f in active]
    out["optional"] = [f for f in schema.get("optional", []) if f in active]
    out["choices"] = {k: v for k, v in schema.get("choices", {}).items() if k in active}
    out["types"] = {k: v for k, v in schema.get("types", {}).items() if k in active}
    out["descriptions"] = {k: v for k, v in schema.get("descriptions", {}).items() if k in active}
    out["defaults"] = {k: v for k, v in schema.get("defaults", {}).items() if k in active}
    out["aliases"] = {k: v for k, v in schema.get("aliases", {}).items() if k in active}
    out["elements"] = {k: v for k, v in schema.get("elements", {}).items() if k in active}
    out["dependencies"] = {k: v for k, v in schema.get("dependencies", {}).items() if k in active}
    out["suboptions"] = {k: v for k, v in schema.get("suboptions", {}).items() if k in active}

    return out