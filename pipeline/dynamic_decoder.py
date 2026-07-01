from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional

from llama_cpp import LlamaGrammar

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


@dataclass
class PiecePlan:
    label: str
    grammar_text: str
    max_tokens: int
    field: Optional[str] = None
    terminal_for_field: bool = True
    fallback_text: Optional[str] = None


@dataclass
class DecodeState:
    query: str
    module_fqn: str
    schema: Dict[str, Any]
    active_fields: List[str]
    value_hints: Dict[str, Any]
    output: str = ""
    pieces: List[PiecePlan] = dataclass_field(default_factory=list)
    generated_fields: List[str] = dataclass_field(default_factory=list)


class StagewiseDynamicDecoder:
    """
    Phase-1 decoder.

    This version keeps the YAML header deterministic and performs
    piece-by-piece grammar rebuilding for the remaining YAML.
    Any piece that has an exact fallback_text is emitted directly,
    which avoids newline/indentation corruption for deterministic syntax.
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

    def _yaml_atom(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value)
        if text and all(c.isalnum() or c in "._/:+-" for c in text):
            return text
        return json.dumps(text)

    def _exact_grammar(self, text: str) -> str:
        return f"root ::= {json.dumps(text)}"

    def _build_single_field_grammar(
        self,
        schema: Dict[str, Any],
        module_fqn: str,
        field: str,
        hint: Any,
    ) -> str:
        projected = project_schema(schema, [field])
        single_hint = {field: hint} if hint is not None else None
        return build_yaml_grammar(
            module_fqn=module_fqn,
            schema=projected,
            include_fields=[field],
            value_hints=single_hint,
            include_header=False,
        )

    def _field_kind(self, spec: FieldSpec, hint: Any) -> str:
        if isinstance(hint, list):
            return "list"
        if isinstance(hint, dict):
            return "dict"

        if spec.type_name in {"list", "array"}:
            return "list"
        if spec.type_name in {"dict", "mapping", "object"} or spec.suboptions:
            return "dict"
        return "scalar"

    def _yaml_header(self, module_fqn: str) -> str:
        return f"- name: Generated Task\n  {module_fqn}:\n"

    def _plan_scalar_piece(
        self,
        schema: Dict[str, Any],
        module_fqn: str,
        field: str,
        hint: Any,
    ) -> PiecePlan:
        grammar = self._build_single_field_grammar(schema, module_fqn, field, hint)
        return PiecePlan(
            label=f"{field}:scalar-line",
            grammar_text=grammar,
            max_tokens=32,
            field=field,
            terminal_for_field=True,
        )

    def _plan_list_pieces(
        self,
        schema: Dict[str, Any],
        module_fqn: str,
        field: str,
        hint: Any,
    ) -> List[PiecePlan]:
        indent_field = "    "
        indent_item = "        "
        pieces: List[PiecePlan] = []

        if isinstance(hint, list) and hint:
            header_text = f"{indent_field}{field}:\n"
            pieces.append(
                PiecePlan(
                    label=f"{field}:list-header",
                    grammar_text=self._exact_grammar(header_text),
                    max_tokens=8,
                    field=field,
                    terminal_for_field=False,
                    fallback_text=header_text,
                )
            )

            for i, item in enumerate(hint):
                item_text = f"{indent_item}- {self._yaml_atom(item)}\n"
                pieces.append(
                    PiecePlan(
                        label=f"{field}:item-{i + 1}",
                        grammar_text=self._exact_grammar(item_text),
                        max_tokens=16,
                        field=field,
                        terminal_for_field=(i == len(hint) - 1),
                        fallback_text=item_text,
                    )
                )
            return pieces

        grammar = self._build_single_field_grammar(schema, module_fqn, field, hint)
        pieces.append(
            PiecePlan(
                label=f"{field}:list-block",
                grammar_text=grammar,
                max_tokens=64,
                field=field,
                terminal_for_field=True,
            )
        )
        return pieces

    def _plan_dict_pieces(
        self,
        schema: Dict[str, Any],
        module_fqn: str,
        field: str,
        hint: Any,
    ) -> List[PiecePlan]:
        indent_field = "    "
        indent_item = "        "
        pieces: List[PiecePlan] = []

        if isinstance(hint, dict) and hint:
            header_text = f"{indent_field}{field}:\n"
            pieces.append(
                PiecePlan(
                    label=f"{field}:dict-header",
                    grammar_text=self._exact_grammar(header_text),
                    max_tokens=8,
                    field=field,
                    terminal_for_field=False,
                    fallback_text=header_text,
                )
            )

            items = list(hint.items())
            for i, (k, v) in enumerate(items):
                pair_text = f"{indent_item}{k}: {self._yaml_atom(v)}\n"
                pieces.append(
                    PiecePlan(
                        label=f"{field}:pair-{i + 1}",
                        grammar_text=self._exact_grammar(pair_text),
                        max_tokens=16,
                        field=field,
                        terminal_for_field=(i == len(items) - 1),
                        fallback_text=pair_text,
                    )
                )
            return pieces

        grammar = self._build_single_field_grammar(schema, module_fqn, field, hint)
        pieces.append(
            PiecePlan(
                label=f"{field}:dict-block",
                grammar_text=grammar,
                max_tokens=64,
                field=field,
                terminal_for_field=True,
            )
        )
        return pieces

    def plan_pieces(
        self,
        query: str,
        schema: Dict[str, Any],
        module_fqn: str,
    ) -> DecodeState:
        trigger_state = infer_active_fields(query, schema, module_fqn=module_fqn)
        active_fields = trigger_state.active_fields or list(schema.get("required", []))
        value_hints = infer_value_hints(query, schema, module_fqn=module_fqn)

        state = DecodeState(
            query=query,
            module_fqn=module_fqn,
            schema=schema,
            active_fields=active_fields,
            value_hints=value_hints,
        )

        ordered_fields = self._ordered_fields(schema, active_fields)
        pieces: List[PiecePlan] = []

        for field in ordered_fields:
            spec = self._field_spec(schema, field)
            hint = value_hints.get(field)
            kind = self._field_kind(spec, hint)

            if kind == "scalar":
                pieces.append(self._plan_scalar_piece(schema, module_fqn, field, hint))
            elif kind == "list":
                pieces.extend(self._plan_list_pieces(schema, module_fqn, field, hint))
            elif kind == "dict":
                pieces.extend(self._plan_dict_pieces(schema, module_fqn, field, hint))
            else:
                pieces.append(self._plan_scalar_piece(schema, module_fqn, field, hint))

        state.pieces = pieces
        return state

    def _generate_piece(self, prompt: str, piece: PiecePlan) -> str:
        # Deterministic syntax pieces should be emitted exactly as planned.
        if piece.fallback_text is not None:
            return piece.fallback_text

        grammar = LlamaGrammar.from_string(piece.grammar_text)
        response = self.llm(
            prompt,
            grammar=grammar,
            temperature=0,
            max_tokens=piece.max_tokens,
            stop=["<|end|>", "<|start|>"],
        )
        text = response["choices"][0]["text"]
        if text is None:
            text = ""
        return text

    def _prompt_for_piece(self, query: str, output: str, piece: PiecePlan) -> str:
        return f"""
You are generating an Ansible YAML task.

User request:
{query}

Already generated YAML:
{output}

Generate only the next YAML piece.
Piece label: {piece.label}
Do not repeat earlier lines.
Do not add markdown.
Do not add code fences.
Do not explain.
Return only the next YAML piece.

YAML:
""".strip()

    def decode(
        self,
        query: str,
        schema: Dict[str, Any],
        module_fqn: str,
        max_tokens: int = 256,
        debug: bool = False,
        return_metadata: bool = False,
    ):
        state = self.plan_pieces(query=query, schema=schema, module_fqn=module_fqn)

        header = self._yaml_header(module_fqn)
        state.output = header

        if debug:
            print("=" * 100)
            print("MODULE_FQN:", module_fqn)
            print("HEADER:")
            print(repr(header))
            print("ACTIVE FIELDS:", state.active_fields)
            print("VALUE HINTS:", state.value_hints)
            print("PIECE COUNT:", len(state.pieces))
            print("PIECES:", [p.label for p in state.pieces])

        if not state.pieces:
            result = state.output.rstrip()
            if return_metadata:
                return result, {
                    "active_fields": state.active_fields,
                    "piece_labels": [],
                    "switch_count": 0,
                    "generated_fields": [],
                    "value_hints": state.value_hints,
                    "module_fqn": module_fqn,
                    "header": header,
                }
            return result

        budget_left = max_tokens

        for idx, piece in enumerate(state.pieces):
            if budget_left <= 0:
                break

            piece_budget = min(piece.max_tokens, budget_left)
            prompt = self._prompt_for_piece(query, state.output, piece)

            if debug:
                print("=" * 100)
                print(f"PIECE {idx + 1}/{len(state.pieces)} :: {piece.label}")
                print("GRAMMAR:")
                print(piece.grammar_text)
                print("OUTPUT SO FAR:")
                print(state.output)

            generated = self._generate_piece(prompt, piece)

            state.output += generated
            budget_left -= piece_budget

            if piece.field and piece.terminal_for_field and piece.field not in state.generated_fields:
                state.generated_fields.append(piece.field)

            if debug:
                print("GENERATED PIECE:")
                print(repr(generated))
                print("FULL OUTPUT:")
                print(state.output)

        result = state.output.rstrip()
        if return_metadata:
            return result, {
                "active_fields": state.active_fields,
                "piece_labels": [p.label for p in state.pieces],
                "switch_count": len(state.pieces),
                "generated_fields": state.generated_fields,
                "value_hints": state.value_hints,
                "module_fqn": module_fqn,
                "header": header,
            }
        return result


def generate_dynamic_yaml(
    query: str,
    schema: Dict[str, Any],
    module_fqn: str,
    max_tokens: int = 256,
    debug: bool = False,
    return_metadata: bool = False,
):
    decoder = StagewiseDynamicDecoder()
    return decoder.decode(
        query=query,
        schema=schema,
        module_fqn=module_fqn,
        max_tokens=max_tokens,
        debug=debug,
        return_metadata=return_metadata,
    )