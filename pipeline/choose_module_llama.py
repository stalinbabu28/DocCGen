from __future__ import annotations

import re
from typing import List, Optional

from pipeline.llm import get_llm

INFO_CUES = (
    " show ",
    " details ",
    " information ",
    " info ",
    " get ",
    " list ",
    " display ",
    " describe ",
)

RESOURCE_HINTS = (
    ("virtual network gateway", "virtualnetworkgateway"),
    ("virtual network", "virtualnetwork"),
    ("network interface", "networkinterface"),
    ("dns zone", "dnszone"),
    ("public ip address", "publicipaddress"),
    ("route table", "routetable"),
    ("storage account", "storageaccount"),
    ("managed disk", "manageddisk"),
    ("load balancer", "loadbalancer"),
    ("application gateway", "appgateway"),
    ("key vault", "keyvault"),
    ("container registry", "containerregistry"),
    ("web app", "webapp"),
    ("aks cluster", "aks"),
    ("sql database", "sqldatabase"),
    ("sql server", "sqlserver"),
    ("subnet", "subnet"),
)


def _module_slug(cand: dict) -> str:
    slug = cand["module_fqn"].split(".")[-1].lower()
    for prefix in ("azure_rm_", "azure_"):
        if slug.startswith(prefix):
            slug = slug[len(prefix):]
    if slug.endswith("_info"):
        slug = slug[:-5]
    return slug


def _query_has_info_cue(query: str) -> bool:
    q = f" {query.lower()} "
    return any(cue in q for cue in INFO_CUES)


def _query_has_resource_hint(query: str) -> bool:
    q = f" {query.lower()} "
    return any(phrase in q for phrase, _ in RESOURCE_HINTS)


def build_prompt(query: str, candidates: List[dict]) -> str:
    lines = []
    for i, cand in enumerate(candidates, 1):
        lines.append(
            f"{i}. {cand['module_fqn']} — {cand.get('short_description', '').strip()}"
        )

    prompt = f"""
You choose the single best Azure Ansible module for a user request.

Return ONLY one exact module name from the candidate list.
Do not explain.
Do not add markdown.
Do not add code fences.
Do not add any extra text.

Prefer the primary resource module over *_info, *instance, *extension, *link, and *group variants unless the user explicitly asks for those concepts.

User request:
{query}

Candidate modules:
{chr(10).join(lines)}

Exact module name:
""".strip()

    return prompt


def _extract_choice(raw: str, candidates: List[dict]) -> Optional[str]:
    raw = (raw or "").strip()

    m = re.match(r"^\s*(\d+)\s*$", raw)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]["module_fqn"]

    for cand in candidates:
        if cand["module_fqn"] in raw:
            return cand["module_fqn"]

    for cand in candidates:
        if re.search(rf"(?<!\w){re.escape(cand['module'])}(?!\w)", raw):
            return cand["module_fqn"]

    for line in raw.splitlines():
        line = line.strip(" \t`'\"-")
        for cand in candidates:
            if line == cand["module_fqn"] or line == cand["module"]:
                return cand["module_fqn"]

    return None


def choose_module(query: str, candidates: List[dict], max_tokens: int = 64) -> dict:
    if not candidates:
        raise ValueError("No candidates provided to choose_module().")

    best = max(
        candidates,
        key=lambda c: (c.get("final_score", c.get("score", 0.0)), c.get("score", 0.0)),
    )
    sorted_scores = sorted(
        (c.get("final_score", c.get("score", 0.0)) for c in candidates),
        reverse=True,
    )
    second_score = sorted_scores[1] if len(sorted_scores) > 1 else float("-inf")

    if _query_has_resource_hint(query) or _query_has_info_cue(query):
        if best.get("final_score", best.get("score", 0.0)) >= second_score + 1.0:
            out = dict(best)
            out["raw_choice_text"] = "heuristic_top_rank"
            out["selected"] = True
            out["selected_method"] = "heuristic"
            return out

    prompt = build_prompt(query, candidates)
    llm = get_llm()

    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0,
        stop=[
            "<|end|>",
            "<|start|>",
            "\n\n",
        ],
    )

    raw = response["choices"][0]["text"]
    choice = _extract_choice(raw, candidates)

    if choice is None:
        choice = candidates[0]["module_fqn"]

    for cand in candidates:
        if cand["module_fqn"] == choice:
            out = dict(cand)
            out["raw_choice_text"] = raw
            out["selected"] = True
            out["selected_method"] = "llm"
            return out

    out = dict(candidates[0])
    out["raw_choice_text"] = raw
    out["selected"] = True
    out["selected_method"] = "fallback"
    return out