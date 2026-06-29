from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from pipeline.llm import get_llm
from pipeline.trigger_rules import infer_active_fields, project_schema
from pipeline.value_hints import infer_value_hints
from pipeline.yaml_grammar_builder import build_yaml_grammar


@dataclass
class FieldSpec:
    name: str
    type_name: str
    choices: list
    suboptions: dict


class StagewiseDynamicDecoder:
    """
    Working intermediate decoder.

    This is not the final paper-faithful parser/backtracking version.
    It generates the YAML in stages:
    1) header
    2) one field block at a time
    3) a fresh grammar is built for each stage
    """

    def __init__(self):
        self.llm = get_llm()

    def _ordered_fields(self, schema: Dict[str, Any], active_fields: List[str]) -> List[str]:
        required = [f for f in schema.get("required", []) if f in active_fields]
        optional = [f for f in schema.get("optional", []) if f in active_fields]

        ordered: List[str] = []
        for field in required + optional + active_fields:
            if field not in ordered:
                ordered.append(field)

        return ordered

    def _field_spec(self, schema: Dict[str, Any], field: str) -> FieldSpec:
        return FieldSpec(
            name=field,
            type_name=str(schema.get("types", {}).get(field, "str")).lower(),
            choices=list(schema.get("choices", {}).get(field, [])),
            suboptions=dict(schema.get("suboptions", {}).get(field, {})),
        )

    def _estimate_field_tokens(self, spec: FieldSpec, hint: Any) -> int:
        if isinstance(hint, list) and hint:
            return 48 + 12 * len(hint)

        if isinstance(hint, dict) and hint:
            return 64 + 16 * len(hint)

        if spec.type_name in {"list", "array"}:
            return 64

        if spec.type_name in {"dict", "mapping", "object"} or spec.suboptions:
            return 80

        return 48

    def _generate_stage(
        self,
        prompt: str,
        grammar_text: str,
        max_tokens: int,
    ) -> str:
        response = self.llm(
            prompt,
            grammar=build_grammar(grammar_text),
            temperature=0,
            max_tokens=max_tokens,
            stop=["<|end|>", "<|start|>"],
        )
        return response["choices"][0]["text"].strip()

    def decode(
        self,
        query: str,
        schema: Dict[str, Any],
        module_fqn: str,
        max_tokens: int = 256,
    ) -> str:
        trigger_state = infer_active_fields(query, schema)
        active_fields = trigger_state.active_fields or list(schema.get("required", []))
        ordered_fields = self._ordered_fields(schema, active_fields)
        value_hints = infer_value_hints(query, schema)

        header = f"- name: Generated Task\n  {module_fqn}:\n"
        output = header

        if not ordered_fields:
            return output.strip()

        per_field_budget = max(32, max_tokens // max(1, len(ordered_fields)))

        for idx, field in enumerate(ordered_fields):
            remaining_fields = ordered_fields[idx:]
            projected = project_schema(schema, [field])

            grammar_text = build_yaml_grammar(
                module_fqn=module_fqn,
                schema=projected,
                include_fields=[field],
                value_hints=value_hints,
                include_header=False,
            )

            spec = self._field_spec(schema, field)
            stage_budget = max(per_field_budget, self._estimate_field_tokens(spec, value_hints.get(field)))

            prompt = f"""
You are generating an Ansible YAML task.

User request:
{query}

Already generated YAML:
{output}

Generate only the next YAML field block.
Do not repeat previous fields.
Do not add markdown.
Do not add code fences.
Return only the YAML for this field.

Next field:
{field}

YAML:
""".strip()

            chunk = self._generate_stage(
                prompt=prompt,
                grammar_text=grammar_text,
                max_tokens=stage_budget,
            )

            if chunk:
                if not chunk.endswith("\n"):
                    chunk += "\n"
                output += chunk

        return output.strip()


def build_grammar(grammar_text: str):
    from llama_cpp import LlamaGrammar
    return LlamaGrammar.from_string(grammar_text)


def generate_dynamic_yaml(
    query: str,
    schema: Dict[str, Any],
    module_fqn: str,
    max_tokens: int = 256,
) -> str:
    decoder = StagewiseDynamicDecoder()
    return decoder.decode(
        query=query,
        schema=schema,
        module_fqn=module_fqn,
        max_tokens=max_tokens,
    )