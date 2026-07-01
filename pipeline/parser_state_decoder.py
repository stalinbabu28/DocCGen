from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_cpp import LlamaGrammar

from pipeline.llm import get_llm
from pipeline.parser_state import FieldKind, ParserMode, ParserState
from pipeline.trigger_rules import infer_active_fields, project_schema
from pipeline.value_hints import infer_value_hints
from pipeline.yaml_grammar_builder import build_yaml_grammar


@dataclass
class TemplatePiece:
    label: str
    grammar_text: str
    max_tokens: int
    kind: str
    direct_text: Optional[str] = None
    field: Optional[str] = None
    terminal_for_field: bool = True


class Phase2ParserStateDecoder:
    """
    Phase 2: parser-state template mutation.

    This decoder mutates templates by parser state:
    - HEADER
    - SCALAR field
    - LIST header -> LIST item(s)
    - DICT header -> DICT pair(s)

    It is still line/block-level, not full token-level backtracking.
    """

    def __init__(self):
        self.llm = get_llm()

    def _yaml_atom(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        text = str(value)
        if text and all(c.isalnum() or c in "._/:+-" for c in text):
            return text
        return json.dumps(text)

    def _exact_grammar(self, text: str) -> str:
        return f"root ::= {json.dumps(text)}"

    def _ordered_fields(self, schema: Dict[str, Any], active_fields: List[str]) -> List[str]:
        required = [f for f in schema.get("required", []) if f in active_fields]
        optional = [f for f in schema.get("optional", []) if f in active_fields]

        ordered: List[str] = []
        for field in required + optional + active_fields:
            if field not in ordered:
                ordered.append(field)
        return ordered

    def _field_kind(self, schema: Dict[str, Any], field: str, hint: Any) -> FieldKind:
        type_name = str(schema.get("types", {}).get(field, "str")).lower()
        suboptions = dict(schema.get("suboptions", {}).get(field, {}))

        if isinstance(hint, list):
            return FieldKind.LIST
        if isinstance(hint, dict):
            return FieldKind.DICT

        if type_name in {"list", "array"}:
            return FieldKind.LIST
        if type_name in {"dict", "mapping", "object"} or suboptions:
            return FieldKind.DICT
        return FieldKind.SCALAR

    def _single_field_grammar(
        self,
        schema: Dict[str, Any],
        module_fqn: str,
        field: str,
        hint: Any,
    ) -> str:
        projected = project_schema(schema, [field])
        value_hints = {field: hint} if hint is not None else None
        return build_yaml_grammar(
            module_fqn=module_fqn,
            schema=projected,
            include_fields=[field],
            value_hints=value_hints,
            include_header=False,
        )

    def _header_piece(self, module_fqn: str) -> TemplatePiece:
        text = f"- name: Generated Task\n  {module_fqn}:\n"
        return TemplatePiece(
            label="header",
            grammar_text=self._exact_grammar(text),
            max_tokens=16,
            kind="header",
            direct_text=text,
            terminal_for_field=False,
        )

    def _scalar_piece(self, state: ParserState) -> TemplatePiece:
        assert state.current is not None
        field = state.current.name
        hint = state.current.hint

        if hint is not None:
            text = f"    {field}: {self._yaml_atom(hint)}\n"
            return TemplatePiece(
                label=f"{field}:scalar-direct",
                grammar_text=self._exact_grammar(text),
                max_tokens=16,
                kind="scalar",
                direct_text=text,
                field=field,
            )

        grammar = self._single_field_grammar(state.schema, state.module_fqn, field, hint)
        return TemplatePiece(
            label=f"{field}:scalar-grammar",
            grammar_text=grammar,
            max_tokens=32,
            kind="scalar",
            field=field,
        )

    def _list_piece(self, state: ParserState) -> TemplatePiece:
        assert state.current is not None
        field = state.current.name
        ctx = state.current

        if not ctx.started:
            # First mutation: enter list template.
            if isinstance(ctx.hint, list) and ctx.hint:
                text = f"    {field}:\n"
                return TemplatePiece(
                    label=f"{field}:list-header",
                    grammar_text=self._exact_grammar(text),
                    max_tokens=8,
                    kind="list-header",
                    direct_text=text,
                    field=field,
                    terminal_for_field=False,
                )

            # Unknown list: fall back to whole-block grammar.
            grammar = self._single_field_grammar(state.schema, state.module_fqn, field, ctx.hint)
            return TemplatePiece(
                label=f"{field}:list-block",
                grammar_text=grammar,
                max_tokens=64,
                kind="list-block",
                field=field,
            )

        # List items after the header.
        if isinstance(ctx.hint, list) and ctx.item_index < len(ctx.hint):
            item = ctx.hint[ctx.item_index]
            text = f"        - {self._yaml_atom(item)}\n"
            return TemplatePiece(
                label=f"{field}:item-{ctx.item_index + 1}",
                grammar_text=self._exact_grammar(text),
                max_tokens=16,
                kind="list-item",
                direct_text=text,
                field=field,
                terminal_for_field=(ctx.item_index == len(ctx.hint) - 1),
            )

        grammar = self._single_field_grammar(state.schema, state.module_fqn, field, ctx.hint)
        return TemplatePiece(
            label=f"{field}:list-block",
            grammar_text=grammar,
            max_tokens=64,
            kind="list-block",
            field=field,
        )

    def _dict_piece(self, state: ParserState) -> TemplatePiece:
        assert state.current is not None
        field = state.current.name
        ctx = state.current

        if not ctx.started:
            # First mutation: enter dict template.
            if isinstance(ctx.hint, dict) and ctx.hint:
                text = f"    {field}:\n"
                return TemplatePiece(
                    label=f"{field}:dict-header",
                    grammar_text=self._exact_grammar(text),
                    max_tokens=8,
                    kind="dict-header",
                    direct_text=text,
                    field=field,
                    terminal_for_field=False,
                )

            grammar = self._single_field_grammar(state.schema, state.module_fqn, field, ctx.hint)
            return TemplatePiece(
                label=f"{field}:dict-block",
                grammar_text=grammar,
                max_tokens=64,
                kind="dict-block",
                field=field,
            )

        if isinstance(ctx.hint, dict) and ctx.pair_index < len(ctx.pairs):
            key, value = ctx.pairs[ctx.pair_index]
            text = f"        {key}: {self._yaml_atom(value)}\n"
            return TemplatePiece(
                label=f"{field}:pair-{ctx.pair_index + 1}",
                grammar_text=self._exact_grammar(text),
                max_tokens=16,
                kind="dict-pair",
                direct_text=text,
                field=field,
                terminal_for_field=(ctx.pair_index == len(ctx.pairs) - 1),
            )

        grammar = self._single_field_grammar(state.schema, state.module_fqn, field, ctx.hint)
        return TemplatePiece(
            label=f"{field}:dict-block",
            grammar_text=grammar,
            max_tokens=64,
            kind="dict-block",
            field=field,
        )

    def _piece_for_state(self, state: ParserState) -> TemplatePiece:
        if state.mode == ParserMode.HEADER:
            return self._header_piece(state.module_fqn)

        if state.current is None:
            return self._header_piece(state.module_fqn)

        if state.current.kind == FieldKind.SCALAR:
            return self._scalar_piece(state)

        if state.current.kind == FieldKind.LIST:
            return self._list_piece(state)

        if state.current.kind == FieldKind.DICT:
            return self._dict_piece(state)

        return self._scalar_piece(state)

    def _advance_state_after_piece(self, state: ParserState, piece: TemplatePiece) -> None:
        ctx = state.current
        if ctx is None:
            if piece.kind == "header":
                state.mode = ParserMode.FIELD_SELECT
            return

        if piece.kind == "header":
            state.mode = ParserMode.FIELD_SELECT
            return

        if piece.kind == "scalar":
            state.finish_current_field()
            return

        if piece.kind == "list-header":
            ctx.started = True
            state.mode = ParserMode.LIST_ITEMS
            return

        if piece.kind == "list-item":
            ctx.item_index += 1
            if isinstance(ctx.hint, list) and ctx.item_index >= len(ctx.hint):
                state.finish_current_field()
            else:
                state.mode = ParserMode.LIST_ITEMS
            return

        if piece.kind == "list-block":
            state.finish_current_field()
            return

        if piece.kind == "dict-header":
            ctx.started = True
            state.mode = ParserMode.DICT_ITEMS
            return

        if piece.kind == "dict-pair":
            ctx.pair_index += 1
            if isinstance(ctx.hint, dict) and ctx.pair_index >= len(ctx.hint):
                state.finish_current_field()
            else:
                state.mode = ParserMode.DICT_ITEMS
            return

        if piece.kind == "dict-block":
            state.finish_current_field()
            return

        state.finish_current_field()

    def _emit_piece(self, query: str, output: str, piece: TemplatePiece) -> str:
        if piece.direct_text is not None:
            return piece.direct_text

        prompt = f"""
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

        grammar = LlamaGrammar.from_string(piece.grammar_text)
        response = self.llm(
            prompt,
            grammar=grammar,
            temperature=0,
            max_tokens=piece.max_tokens,
            stop=["<|end|>", "<|start|>"],
        )
        text = response["choices"][0]["text"]
        return text or ""

    def decode(
        self,
        query: str,
        schema: Dict[str, Any],
        module_fqn: str,
        max_tokens: int = 256,
        debug: bool = False,
        return_metadata: bool = False,
    ):
        trigger_state = infer_active_fields(query, schema, module_fqn=module_fqn)
        active_fields = trigger_state.active_fields or list(schema.get("required", []))
        value_hints = infer_value_hints(query, schema, module_fqn=module_fqn)
        ordered_fields = self._ordered_fields(schema, active_fields)

        state = ParserState(
            query=query,
            module_fqn=module_fqn,
            schema=schema,
            active_fields=active_fields,
            value_hints=value_hints,
            ordered_fields=ordered_fields,
            mode=ParserMode.HEADER,
        )

        header = f"- name: Generated Task\n  {module_fqn}:\n"
        state.output = header
        state.mark_emitted("header")
        state.mode = ParserMode.FIELD_SELECT

        if debug:
            print("=" * 100)
            print("MODULE_FQN:", module_fqn)
            print("HEADER:")
            print(repr(header))
            print("ACTIVE FIELDS:", state.active_fields)
            print("VALUE HINTS:", state.value_hints)
            print("ORDERED FIELDS:", state.ordered_fields)

        budget_left = max_tokens

        while budget_left > 0 and not state.done():
            if state.current is None:
                if not state.has_more_fields():
                    state.mode = ParserMode.DONE
                    break

                field_name = state.ordered_fields[state.field_index]
                hint = state.value_hints.get(field_name)
                suboptions = dict(state.schema.get("suboptions", {}).get(field_name, {}))
                kind = self._field_kind(state.schema, field_name, hint)
                state.select_current_field(kind=kind, hint=hint, suboptions=suboptions)

            piece = self._piece_for_state(state)

            if debug:
                print("=" * 100)
                print(f"STATE FIELD: {state.current_field_name()}")
                print(f"PIECE: {piece.label}")
                print("GRAMMAR:")
                print(piece.grammar_text)
                print("OUTPUT SO FAR:")
                print(state.output)

            generated = self._emit_piece(query, state.output, piece)
            state.output += generated
            state.mark_emitted(piece.label)
            budget_left -= piece.max_tokens

            if debug:
                print("GENERATED PIECE:")
                print(repr(generated))
                print("FULL OUTPUT:")
                print(state.output)

            self._advance_state_after_piece(state, piece)

        result = state.output.rstrip()

        if return_metadata:
            return result, {
                "module_fqn": module_fqn,
                "active_fields": state.active_fields,
                "ordered_fields": state.ordered_fields,
                "value_hints": state.value_hints,
                "transition_log": state.transition_log,
                "switch_count": len([x for x in state.transition_log if x.startswith("emit:")]),
                "output": result,
            }

        return result


def generate_parser_state_yaml(
    query: str,
    schema: Dict[str, Any],
    module_fqn: str,
    max_tokens: int = 256,
    debug: bool = False,
    return_metadata: bool = False,
):
    decoder = Phase2ParserStateDecoder()
    return decoder.decode(
        query=query,
        schema=schema,
        module_fqn=module_fqn,
        max_tokens=max_tokens,
        debug=debug,
        return_metadata=return_metadata,
    )