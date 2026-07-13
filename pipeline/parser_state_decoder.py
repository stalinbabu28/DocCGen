from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_cpp import LlamaGrammar

from pipeline.backtracking import save_checkpoint, restore_checkpoint
from pipeline.decoder_time_triggers import DecoderTimeTriggerEngine, StructuralTriggerKind
from pipeline.llm import get_llm
from pipeline.parser_state import FieldKind, ParserMode, ParserState
from pipeline.trigger_rules import infer_active_fields, project_schema
from pipeline.value_hints import infer_value_hints, placeholder_for_field
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


class Phase3ParserStateDecoder:
    """
    Phase 3 decoder with structural triggers and Phase 4 backtracking support.
    """

    def __init__(self):
        self._llm = None
        self.trigger_engine = DecoderTimeTriggerEngine()

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _yaml_atom(self, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"

        text = str(value)
        if text and all(c.isalnum() or c in "._/:+-" for c in text):
            return text
        return json.dumps(text)

    def _exact_grammar(self, text: str) -> str:
        return f'root ::= {json.dumps(text)}'

    def _ordered_fields(self, schema: Dict[str, Any], active_fields: List[str]) -> List[str]:
        required = list(schema.get("required", []))
        ordered: List[str] = []

        for field in required + active_fields:
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

    def _scalar_piece(self, state: ParserState, loosen: bool = False) -> TemplatePiece:
        assert state.current is not None
        field = state.current.name
        hint = state.current.hint

        if hint is not None and not loosen:
            text = f"    {field}: {self._yaml_atom(hint)}\n"
            return TemplatePiece(
                label=f"{field}:scalar-direct",
                grammar_text=self._exact_grammar(text),
                max_tokens=16,
                kind="scalar-direct",
                direct_text=text,
                field=field,
            )

        grammar_hint = None if loosen else hint
        grammar = self._single_field_grammar(
            state.schema,
            state.module_fqn,
            field,
            grammar_hint,
        )
        return TemplatePiece(
            label=f"{field}:scalar-grammar",
            grammar_text=grammar,
            max_tokens=32,
            kind="scalar",
            field=field,
        )

    def _list_piece(self, state: ParserState, loosen: bool = False) -> TemplatePiece:
        assert state.current is not None
        field = state.current.name
        ctx = state.current

        if not loosen:
            if not ctx.started and isinstance(ctx.hint, list) and ctx.hint:
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

            if ctx.started and isinstance(ctx.hint, list) and ctx.item_index < len(ctx.hint):
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

        grammar = self._single_field_grammar(
            state.schema,
            state.module_fqn,
            field,
            None if loosen else ctx.hint,
        )
        return TemplatePiece(
            label=f"{field}:list-block",
            grammar_text=grammar,
            max_tokens=64,
            kind="list-block",
            field=field,
        )

    def _dict_piece(self, state: ParserState, loosen: bool = False) -> TemplatePiece:
        assert state.current is not None
        field = state.current.name
        ctx = state.current

        if not loosen:
            if not ctx.started and isinstance(ctx.hint, dict) and ctx.hint:
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

            if ctx.started and isinstance(ctx.hint, dict) and ctx.pair_index < len(ctx.hint):
                key, value = list(ctx.hint.items())[ctx.pair_index]
                text = f"        {key}: {self._yaml_atom(value)}\n"
                return TemplatePiece(
                    label=f"{field}:pair-{ctx.pair_index + 1}",
                    grammar_text=self._exact_grammar(text),
                    max_tokens=16,
                    kind="dict-pair",
                    direct_text=text,
                    field=field,
                    terminal_for_field=(ctx.pair_index == len(ctx.hint) - 1),
                )

        grammar = self._single_field_grammar(
            state.schema,
            state.module_fqn,
            field,
            None if loosen else ctx.hint,
        )
        return TemplatePiece(
            label=f"{field}:dict-block",
            grammar_text=grammar,
            max_tokens=64,
            kind="dict-block",
            field=field,
        )

    def _piece_for_state(self, state: ParserState, loosened_fields: set[str]) -> TemplatePiece:
        if state.mode == ParserMode.HEADER:
            return self._header_piece(state.module_fqn)

        if state.current is None:
            raise RuntimeError("Parser state has no current field while not in HEADER mode.")

        loosen = state.current.name in loosened_fields

        if state.current.kind == FieldKind.SCALAR:
            return self._scalar_piece(state, loosen=loosen)

        if state.current.kind == FieldKind.LIST:
            return self._list_piece(state, loosen=loosen)

        if state.current.kind == FieldKind.DICT:
            return self._dict_piece(state, loosen=loosen)

        return self._scalar_piece(state, loosen=loosen)

    def _emit_piece(self, query: str, output: str, piece: TemplatePiece) -> str:
        if piece.direct_text is not None:
            return piece.direct_text

        llm = self._get_llm()

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
        response = llm(
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
        enable_backtracking: bool = True,
        max_backtracks: int = 12,
    ):
        trigger_state = infer_active_fields(query, schema, module_fqn=module_fqn)
        inferred_fields = list(trigger_state.active_fields or [])

        real_value_hints = infer_value_hints(
            query,
            schema,
            module_fqn=module_fqn,
            include_placeholders=False,
        )

        required_fields = list(schema.get("required", []))

        active_fields: List[str] = []
        for field in required_fields + inferred_fields + list(real_value_hints.keys()):
            if field not in active_fields:
                active_fields.append(field)

        ordered_fields = self._ordered_fields(schema, active_fields)

        value_hints = real_value_hints.copy()
        for field in active_fields:
            if field not in value_hints:
                ph = placeholder_for_field(field)
                if ph is not None:
                    value_hints[field] = ph

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
        state.record_trigger_event("fire:header_emitted")
        state.switch_template(f"module:{module_fqn}")
        state.mode = ParserMode.FIELD_SELECT

        if debug:
            print("=" * 100)
            print("MODULE_FQN:", module_fqn)
            print("HEADER:")
            print(repr(header))
            print("ACTIVE FIELDS:", state.active_fields)
            print("VALUE HINTS:", state.value_hints)
            print("ORDERED FIELDS:", state.ordered_fields)
            print("CURRENT TEMPLATE:", state.current_template())

        budget_left = max_tokens
        piece_index = 0
        backtracks_used = 0
        loosened_fields: set[str] = set()
        piece_attempts: dict[str, int] = {}

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

            piece = self._piece_for_state(state, loosened_fields=loosened_fields)
            checkpoint = save_checkpoint(
                state=state,
                budget_left=budget_left,
                piece_index=piece_index,
                loosened_fields=loosened_fields,
                backtracks_used=backtracks_used,
                metadata={
                    "piece_label": piece.label,
                    "field": piece.field,
                },
            )

            if debug:
                print("=" * 100)
                print(f"STATE FIELD: {state.current_field_name()}")
                print(f"PIECE: {piece.label}")
                print("CURRENT TEMPLATE:", state.current_template())
                print("GRAMMAR:")
                print(piece.grammar_text)
                print("OUTPUT SO FAR:")
                print(state.output)

            try:
                generated = self._emit_piece(query, state.output, piece)

                if generated is None:
                    generated = ""

                if not generated.strip():
                    raise RuntimeError(f"Empty generation for piece {piece.label}")

                state.output += generated
                state.mark_emitted(piece.label)
                budget_left -= piece.max_tokens

                events = self.trigger_engine.apply(
                    state=state,
                    piece_kind=piece.kind,
                    piece_label=piece.label,
                    generated=generated,
                    debug=debug,
                )

                if any(e.kind == StructuralTriggerKind.INVALID_INDENTATION for e in events):
                    raise RuntimeError(
                        f"invalid indentation for {piece.label}: "
                        f"{[e.reason for e in events if e.kind == StructuralTriggerKind.INVALID_INDENTATION]}"
                    )

                if piece.terminal_for_field:
                    state.finish_current_field()

                if debug:
                    print("GENERATED PIECE:")
                    print(repr(generated))
                    print("TRIGGER EVENTS:")
                    print([e.kind.name for e in events])
                    print("FULL OUTPUT:")
                    print(state.output)
                    print("PARSER LOG:")
                    print(state.parser_log)
                    print("DECODER LOG:")
                    print(state.decoder_log)
                    print("TRIGGER LOG:")
                    print(state.trigger_log)
                    print("TEMPLATE STACK:")
                    print(state.template_stack)

                piece_index += 1

            except Exception as exc:
                if debug:
                    print(f"BACKTRACK CANDIDATE: {piece.label} -> {type(exc).__name__}: {exc}")

                if not enable_backtracking:
                    raise

                if backtracks_used >= max_backtracks:
                    state.record_decoder_event(f"backtrack:budget_exhausted:{piece.label}")
                    break

                restored = restore_checkpoint(checkpoint)
                state = restored[0]
                budget_left = restored[1]
                piece_index = restored[2]
                loosened_fields = restored[3]
                backtracks_used = restored[4]
                metadata = restored[5]

                field_key = metadata.get("field") or metadata.get("piece_label") or piece.label
                loosened_fields.add(field_key)
                piece_attempts[field_key] = piece_attempts.get(field_key, 0) + 1
                backtracks_used += 1

                state.record_decoder_event(
                    f"backtrack:{field_key}:{backtracks_used}:{type(exc).__name__}"
                )

                if debug:
                    print(f"RESTORED CHECKPOINT FOR: {field_key}")
                    print("LOOSENED FIELDS:", sorted(loosened_fields))
                    print("BACKTRACKS USED:", backtracks_used)

                continue

        result = state.output.rstrip()

        if return_metadata:
            return result, {
                "module_fqn": module_fqn,
                "active_fields": state.active_fields,
                "ordered_fields": state.ordered_fields,
                "value_hints": state.value_hints,
                "parser_log": state.parser_log,
                "decoder_log": state.decoder_log,
                "trigger_log": state.trigger_log,
                "transition_log": state.transition_log,
                "template_stack": state.template_stack,
                "current_indent": state.current_indent,
                "switch_count": len([x for x in state.trigger_log if x.startswith("fire:")]),
                "output": result,
                "backtracks_used": backtracks_used,
                "loosened_fields": sorted(loosened_fields),
                "piece_attempts": piece_attempts,
            }

        return result


Phase2ParserStateDecoder = Phase3ParserStateDecoder


def generate_parser_state_yaml(
    query: str,
    schema: Dict[str, Any],
    module_fqn: str,
    max_tokens: int = 256,
    debug: bool = False,
    return_metadata: bool = False,
    enable_backtracking: bool = True,
    max_backtracks: int = 12,
):
    decoder = Phase3ParserStateDecoder()
    return decoder.decode(
        query=query,
        schema=schema,
        module_fqn=module_fqn,
        max_tokens=max_tokens,
        debug=debug,
        return_metadata=return_metadata,
        enable_backtracking=enable_backtracking,
        max_backtracks=max_backtracks,
    )