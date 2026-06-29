from __future__ import annotations

import re
from typing import Any, Dict, Optional

NAME_RE = r"[A-Za-z0-9._/-]+(?:\.[A-Za-z0-9._/-]+)*"
IPV4_RE = r"(?:\d{1,3}\.){3}\d{1,3}"

STOPWORDS = {"and", "with", "tags", "tag", "append", "true", "false", "yes", "no"}


def _first_match(text: str, patterns: list[str]) -> Optional[str]:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" \t\r\n.,;:()[]{}")
    return None


def _normalize_bool(value: str) -> str:
    v = value.strip().lower()
    if v in {"true", "yes", "1"}:
        return "true"
    if v in {"false", "no", "0"}:
        return "false"
    return value


def _extract_ipv4_list(query: str) -> list[str]:
    ips = re.findall(IPV4_RE, query)
    out = []
    for ip in ips:
        if ip not in out:
            out.append(ip)
    return out


def _extract_tags_dict(query: str) -> Dict[str, Any]:
    q = query.strip()
    if re.search(r"\bappend\s+tags\b", q, flags=re.IGNORECASE):
        # append-tags-only requests should not create a fake tags dict
        if not re.search(r"\b(?:with\s+)?tags?\b.*\b[A-Za-z0-9._/-]+\s+[A-Za-z0-9._/-]+\b", q, flags=re.IGNORECASE | re.DOTALL):
            return {}

    m = re.search(r"\b(?:with\s+)?tags?\b(.*)$", q, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return {}

    segment = m.group(1)

    # Stop when another clause starts.
    segment = re.split(r"\band\s+append\s+tags\b", segment, maxsplit=1, flags=re.IGNORECASE)[0]
    segment = re.split(r"\band\s+dns\s+servers\b", segment, maxsplit=1, flags=re.IGNORECASE)[0]
    segment = re.split(r"\band\s+location\b", segment, maxsplit=1, flags=re.IGNORECASE)[0]
    segment = re.split(r"\band\s+resource\s+group\b", segment, maxsplit=1, flags=re.IGNORECASE)[0]

    kv_pairs = re.findall(r"([A-Za-z0-9._/-]+)\s*[:=]\s*([A-Za-z0-9._/-]+)", segment)
    if kv_pairs:
        return {k: v for k, v in kv_pairs}

    tokens = re.findall(r"[A-Za-z0-9._/-]+", segment)
    tokens = [t for t in tokens if t.lower() not in STOPWORDS]

    if len(tokens) < 4:
        return {}

    out: Dict[str, Any] = {}
    i = 0
    while i + 1 < len(tokens):
        key = tokens[i]
        value = tokens[i + 1]
        if key.lower() not in STOPWORDS and value.lower() not in STOPWORDS:
            out[key] = value
        i += 2

    return out


def _resource_name_patterns() -> list[str]:
    return [
        rf"\bdelete\s+subnet\s+({NAME_RE})",
        rf"\b(?:create|delete|show|get|list|update)\s+a?\s*subnet\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bvirtual network gateway\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bvirtual network\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bdns zone\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bpublic ip address\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\broute table\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bstorage account\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bmanaged disk\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bload balancer\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bapplication gateway\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bweb app\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bnetwork interface\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bvirtual machine scale set\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bvirtual machine\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\baks cluster\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bcontainer registry\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bkey vault\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bsql database\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bsql server\s+(?:called\s+|named\s+)?({NAME_RE})",
        rf"\bnamed\s+({NAME_RE})",
        rf"\bcalled\s+({NAME_RE})",
    ]


def infer_value_hints(query: str, schema: Dict[str, Any], module_fqn: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract exact literals from the user query for fields that are easy to ground.
    Returns scalars plus structured hints for lists/dicts.
    """
    hints: Dict[str, Any] = {}
    q = query.strip()
    q_lower = f" {q.lower()} "
    info_module = bool(module_fqn and module_fqn.endswith("_info"))
    types = schema.get("types", {}) or {}

    if "state" in types and (" delete " in q_lower or " remove " in q_lower):
        hints["state"] = "absent"

    if "append_tags" in types:
        m = re.search(r"\bappend\s+tags\s+(true|false|yes|no)\b", q, flags=re.IGNORECASE)
        if m:
            hints["append_tags"] = _normalize_bool(m.group(1))

    if "resource_group" in types:
        rg = _first_match(q, [rf"\bresource group\s+({NAME_RE})"])
        if rg:
            hints["resource_group"] = rg

    if "name" in types:
        name = _first_match(q, _resource_name_patterns())
        if name:
            hints["name"] = name

    if "subnet_name" in types:
        subnet = _first_match(
            q,
            [
                rf"\bdelete\s+subnet\s+({NAME_RE})",
                rf"\b(?:create|delete|show|get|list|update)\s+a?\s*subnet\s+(?:called\s+|named\s+)?({NAME_RE})",
                rf"\bsubnet\s+(?:called\s+|named\s+)?({NAME_RE})",
            ],
        )
        if subnet:
            hints["subnet_name"] = subnet

    if "virtual_network_name" in types:
        vnet = _first_match(
            q,
            [
                rf"\b(?:in|on|with|to|attached to)\s+virtual network\s+({NAME_RE})",
                rf"\bvirtual network\s+(?:called\s+|named\s+)?({NAME_RE})",
            ],
        )
        if vnet and vnet.lower() != "gateway":
            hints["virtual_network_name"] = vnet

    if "virtual_network" in types:
        vnet = _first_match(
            q,
            [
                rf"\b(?:in|on|with|to|attached to)\s+virtual network\s+({NAME_RE})",
            ],
        )
        if vnet and vnet.lower() != "gateway":
            hints["virtual_network"] = vnet

    if "vault_name" in types:
        vault = _first_match(q, [rf"\bkey vault\s+(?:called\s+|named\s+)?({NAME_RE})", rf"\bvault\s+(?:called\s+|named\s+)?({NAME_RE})"])
        if vault:
            hints["vault_name"] = vault

    if "dns_servers" in types and re.search(r"\bdns\s+servers?\b", q, flags=re.IGNORECASE):
        ips = _extract_ipv4_list(q)
        if ips:
            hints["dns_servers"] = ips

    if "tags" in types:
        tags = _extract_tags_dict(q)
        if tags:
            hints["tags"] = tags

    # Conservative fallback for other non-sensitive fields.
    for field, type_name in types.items():
        if field in hints or field in {"name", "resource_group", "subnet_name", "virtual_network_name", "virtual_network", "vault_name", "tags", "dns_servers", "state", "append_tags"}:
            continue

        if info_module:
            continue

        phrase = field.replace("_", " ")
        if type_name in {"bool", "boolean"}:
            m = re.search(rf"\b{re.escape(phrase)}\s+(true|false|yes|no)\b", q, flags=re.IGNORECASE)
            if m:
                hints[field] = _normalize_bool(m.group(1))
                continue

        m = re.search(
            rf"\b{re.escape(phrase)}\s+(?:called|named|with|of|to|from|in|for)\s+({NAME_RE})",
            q,
            flags=re.IGNORECASE,
        )
        if not m:
            m = re.search(rf"\b{re.escape(phrase)}\s+({NAME_RE})", q, flags=re.IGNORECASE)
        if m:
            hints[field] = m.group(1).strip(" \t\r\n.,;:()[]{}")

    return hints