from __future__ import annotations

import json
from functools import lru_cache
from typing import List, Optional, Tuple

from llama_cpp import Llama

MODEL_PATH = "/home/dao-lab/.lmstudio/models/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-F16.gguf"

IGNORED_KEYS = {
    "ad_user",
    "adfs_authority_url",
    "api_profile",
    "auth_source",
    "cert_validation_mode",
    "client_id",
    "cloud_environment",
    "log_mode",
    "log_path",
    "password",
    "profile",
    "secret",
    "subscription_id",
    "tenant",
    "thumbprint",
    "x509_certificate_path",
}

IMPORTANT_OPTIONALS = {
    "name",
    "resource_group",
    "virtual_network_name",
    "state",
    "vault_name",
    "show_kubeconfig",
    "show_blob_cors",
    "show_connection_string",
}


@lru_cache(maxsize=1)
def get_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=20,
        verbose=False,
    )


def _desc(meta) -> str:
    desc = meta.get("description", "")
    if isinstance(desc, list):
        desc = " ".join(desc)
    return " ".join(str(desc).split())


def _choices(meta) -> str:
    choices = meta.get("choices")
    if not choices:
        return ""
    if isinstance(choices, list):
        return ", ".join(map(str, choices))
    return str(choices)


def _load_doc(doc_path: str) -> dict:
    with open(doc_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _field_lines(doc: dict, query: str) -> Tuple[List[str], List[str]]:
    options = doc.get("options", {}) or {}
    q = query.lower()

    required_lines: List[str] = []
    optional_lines: List[str] = []

    for name, meta in options.items():
        if not isinstance(meta, dict):
            continue
        if name in IGNORED_KEYS:
            continue

        desc = _desc(meta)
        choice_text = _choices(meta)

        line = f"- {name}"
        if desc:
            line += f": {desc}"
        if choice_text:
            line += f" (choices: {choice_text})"

        if meta.get("required", False):
            required_lines.append(line)
        else:
            keep = False

            if name in IMPORTANT_OPTIONALS:
                keep = True

            if name in q:
                keep = True

            if name.replace("_", " ") in q:
                keep = True

            if keep:
                optional_lines.append(line)

    # Fallback: include a few non-auth optional keys so the model still sees
    # useful schema context, but keep the list short.
    if not optional_lines:
        for name, meta in options.items():
            if not isinstance(meta, dict):
                continue
            if name in IGNORED_KEYS or meta.get("required", False):
                continue

            desc = _desc(meta)
            choice_text = _choices(meta)

            line = f"- {name}"
            if desc:
                line += f": {desc}"
            if choice_text:
                line += f" (choices: {choice_text})"

            optional_lines.append(line)
            if len(optional_lines) >= 12:
                break

    # Keep state available for delete/remove requests only.
    if ("delete" in q or "remove" in q or "absent" in q) and "state" in options:
        state_meta = options["state"]
        state_desc = _desc(state_meta)
        choice_text = _choices(state_meta)
        line = "- state"
        if state_desc:
            line += f": {state_desc}"
        if choice_text:
            line += f" (choices: {choice_text})"
        if not any(x.startswith("- state") for x in required_lines + optional_lines):
            optional_lines.insert(0, line)

    return required_lines, optional_lines


def build_prompt(query: str, doc: dict) -> str:
    required_lines, optional_lines = _field_lines(doc, query)

    module = doc.get("module", "")
    short_desc = doc.get("short_description", "")

    prompt = f"""
You extract Azure module parameters from a user request.

Return ONLY one JSON object.
Do not explain.
Do not add markdown.
Do not add code fences.
Do not add any extra text before or after the JSON.
Do not guess missing values.
Do not abbreviate or truncate exact values.
Copy exact substrings from the user request.
If a value is not explicitly present, omit the field.
For delete/remove requests, set state to "absent" when the schema supports it.
For create/show/list requests, do not add state unless the request explicitly mentions it.

User request:
{query}

Module:
{module}

Short description:
{short_desc}

Required parameters:
{chr(10).join(required_lines) if required_lines else "(none)"}

Relevant optional parameters:
{chr(10).join(optional_lines) if optional_lines else "(none)"}

Output JSON only:
""".strip()

    return prompt


def generate_raw_parameters(
    query: str,
    doc_path: str,
    schema: Optional[dict] = None,
    max_tokens: int = 256,
):
    """
    Returns raw model text.
    Uses doc_path because we want descriptions and aliases from the full doc.
    """
    doc = _load_doc(doc_path)
    prompt = build_prompt(query, doc)

    llm = get_llm()

    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0,
        stop=[
            "<|end|>",
            "<|start|>",
        ],
    )

    return response["choices"][0]["text"]