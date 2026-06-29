from __future__ import annotations

import json
from functools import lru_cache

from llama_cpp import Llama, LlamaGrammar

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


def run_case(title: str, query: str, schema: dict, include_fields: list[str], max_tokens: int = 96):
    print("\n" + "=" * 80)
    print(title)

    grammar_text = build_grammar(schema, include_fields=include_fields)

    print("\nGRAMMAR:")
    print(grammar_text)
    print("=" * 80)

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
        max_tokens=max_tokens,
        stop=[
            "<|end|>",
            "<|start|>",
        ],
    )

    raw = response["choices"][0]["text"].strip()

    print("\nRAW OUTPUT:")
    print(raw)

    try:
        parsed = json.loads(raw)
        print("\nPARSED JSON:")
        print(parsed)
    except Exception as e:
        print("\nJSON PARSE FAILED:")
        print(repr(e))


def main():
    # ENUM
    enum_schema = {
        "required": ["state"],
        "optional": [],
        "choices": {
            "state": ["present", "absent"],
        },
        "types": {
            "state": "str",
        },
    }
    run_case(
        title="ENUM TEST",
        query="delete the subnet so state should be absent",
        schema=enum_schema,
        include_fields=["state"],
    )

    # BOOL
    bool_schema = {
        "required": ["append_tags"],
        "optional": [],
        "choices": {},
        "types": {
            "append_tags": "bool",
        },
    }
    run_case(
        title="BOOL TEST",
        query="set append tags to true",
        schema=bool_schema,
        include_fields=["append_tags"],
    )

    # LIST
    list_schema = {
        "required": ["address_prefixes_cidr"],
        "optional": [],
        "choices": {},
        "types": {
            "address_prefixes_cidr": "list",
        },
    }
    run_case(
        title="LIST TEST",
        query="set address prefixes to 10.0.0.0/16",
        schema=list_schema,
        include_fields=["address_prefixes_cidr"],
    )

    # DICT
    dict_schema = {
        "required": ["tags"],
        "optional": [],
        "choices": {},
        "types": {
            "tags": "dict",
        },
    }
    run_case(
        title="DICT TEST",
        query="set tags env prod",
        schema=dict_schema,
        include_fields=["tags"],
    )


if __name__ == "__main__":
    main()