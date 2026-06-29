from __future__ import annotations

from copy import deepcopy
from functools import lru_cache

from llama_cpp import Llama, LlamaGrammar

from pipeline.field_policy import select_fields
from pipeline.grammar_builder import build_grammar

MODEL_PATH = (
    "/home/dao-lab/.lmstudio/models/"
    "unsloth/gpt-oss-20b-GGUF/"
    "gpt-oss-20b-F16.gguf"
)


@lru_cache(maxsize=1)
def get_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=20,
        verbose=False,
    )


def _is_delete_request(query: str) -> bool:
    q = query.lower()
    return "delete" in q or "remove" in q


def generate_constrained_json(query, schema):
    """
    Generate JSON under a schema-derived grammar.
    """
    fields = select_fields(query, schema)

    if not fields:
        fields = list(schema.get("required", []))

    # For delete/remove requests, force state to absent when state is in scope.
    schema_for_grammar = deepcopy(schema)
    if _is_delete_request(query) and "state" in fields:
        schema_for_grammar.setdefault("choices", {})
        schema_for_grammar["choices"]["state"] = ["absent"]

    grammar_text = build_grammar(
        schema_for_grammar,
        include_fields=fields,
    )

    grammar = LlamaGrammar.from_string(grammar_text)

    prompt = f"""
Extract Azure module parameters.

Return only JSON.

User request:
{query}

JSON:
""".strip()

    llm = get_llm()

    response = llm(
        prompt,
        grammar=grammar,
        temperature=0,
        max_tokens=128,
        stop=[
            "<|end|>",
            "<|start|>",
        ],
    )

    return response["choices"][0]["text"].strip()