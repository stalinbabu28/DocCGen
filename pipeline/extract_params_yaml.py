from __future__ import annotations

import os

from llama_cpp import LlamaGrammar

from pipeline.dynamic_decoder import generate_dynamic_yaml
from pipeline.parser_state_decoder import generate_parser_state_yaml
from pipeline.trigger_rules import infer_active_fields, project_schema
from pipeline.value_hints import infer_value_hints
from pipeline.yaml_grammar_builder import build_yaml_grammar
from pipeline.llm import get_llm


def _is_delete_request(query: str) -> bool:
    q = query.lower()
    return "delete" in q or "remove" in q


def generate_constrained_yaml(
    query: str,
    schema: dict,
    module_fqn: str,
    max_tokens: int = 128,
) -> str:
    if os.getenv("USE_PARSER_STATE_DECODER", "0") == "1":
        return generate_parser_state_yaml(
            query=query,
            schema=schema,
            module_fqn=module_fqn,
            max_tokens=max_tokens,
        )

    if os.getenv("USE_DYNAMIC_DECODER", "0") == "1":
        return generate_dynamic_yaml(
            query=query,
            schema=schema,
            module_fqn=module_fqn,
            max_tokens=max_tokens,
        )

    trigger_state = infer_active_fields(query, schema, module_fqn=module_fqn)
    fields = trigger_state.active_fields

    if not fields:
        fields = list(schema.get("required", []))

    schema_for_grammar = project_schema(schema, fields)
    value_hints = infer_value_hints(query, schema, module_fqn=module_fqn)

    if _is_delete_request(query) and "state" in fields:
        schema_for_grammar.setdefault("choices", {})
        schema_for_grammar["choices"]["state"] = ["absent"]
        value_hints["state"] = "absent"

    grammar_text = build_yaml_grammar(
        module_fqn=module_fqn,
        schema=schema_for_grammar,
        include_fields=fields,
        value_hints=value_hints,
        include_header=True,
    )

    grammar = LlamaGrammar.from_string(grammar_text)

    prompt = f"""
Extract the YAML task only.

Return only YAML.
Do not explain.
Do not add markdown.
Do not add code fences.
Do not add any extra text.

User request:
{query}

YAML:
""".strip()

    llm = get_llm()

    response = llm(
        prompt,
        grammar=grammar,
        temperature=0,
        max_tokens=max_tokens,
        stop=["<|end|>", "<|start|>"],
    )

    return response["choices"][0]["text"].strip()