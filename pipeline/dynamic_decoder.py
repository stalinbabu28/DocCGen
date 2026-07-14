from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field as dataclass_field
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
    emitted_fields: List[str] = dataclass_field(default_factory=list)
    output: str = ""


class StagewiseDynamicDecoder:
    """
    Dynamic field-by-field decoder.

    Behavior:
    - required fields are always kept
    - active fields are recomputed after each accepted field
    - grammar is rebuilt from the current remaining schema branch
    - each field gets retry attempts with looser decoding on failure
    """

    def __init__(self):
        self.llm = get_llm()
        self.trigger_engine = DecoderTimeTriggerEngine()

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

    def _yaml_header(self, module_fqn: str) -> str:
        return f"- name: Generated Task\n  {module_fqn}:\n"

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

    def _compute_plan(
        self,
        query: str,
        schema: Dict[str, Any],
        module_fqn: str,
        emitted_fields: List[str],
    ) -> tuple[List[str], List[str], Dict[str, Any]]:
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
            if field in emitted_fields:
                continue
            if field not in active_fields:
                active_fields.append(field)

        ordered_fields: List[str] = []
        for field in required_fields + active_fields:
            if field in emitted_fields:
                continue
            if field not in ordered_fields:
                ordered_fields.append(field)

        value_hints: Dict[str, Any] = dict(real_value_hints)
        for field in active_fields:
            if field not in value_hints:
                ph = placeholder_for_field(field)
                if ph is not None:
                    value_hints[field] = ph

        return active_fields, ordered_fields, value_hints

    def _piece_for_field(
        self,
        state: DecodeState,
        schema: Dict[str, Any],
        module_fqn: str,
        field: str,
        attempt: int,
    ) -> PiecePlan:
        spec = self._field_spec(schema, field)
        hint = state.value_hints.get(field)
        kind = self._field_kind(spec, hint)

        if kind == "scalar":
            if attempt == 0 and hint is not None:
                text = f"    {field}: {self._yaml_atom(hint)}\n"
                return PiecePlan(
                    label=f"{field}:scalar-direct",
                    grammar_text=self._exact_grammar(text),
                    max_tokens=16,
                    field=field,
                    terminal_for_field=True,
                    fallback_text=text,
                )

            if attempt >= 2:
                fallback = placeholder_for_field(field)
                if fallback is not None:
                    text = f"    {field}: {self._yaml_atom(fallback)}\n"
                    return PiecePlan(
                        label=f"{field}:scalar-fallback",
                        grammar_text=self._exact_grammar(text),
                        max_tokens=16,
                        field=field,
                        terminal_for_field=True,
                        fallback_text=text,
                    )

            grammar = self._build_single_field_grammar(
                schema=schema,
                module_fqn=module_fqn,
                field=field,
                hint=None if attempt > 0 else hint,
            )
            return PiecePlan(
                label=f"{field}:scalar-grammar",
                grammar_text=grammar,
                max_tokens=32,
                field=field,
                terminal_for_field=True,
            )

        grammar_hint = hint if attempt == 0 else None
        grammar = self._build_single_field_grammar(
            schema=schema,
            module_fqn=module_fqn,
            field=field,
            hint=grammar_hint,
        )
        return PiecePlan(
            label=f"{field}:{kind}-grammar",
            grammar_text=grammar,
            max_tokens=64,
            field=field,
            terminal_for_field=True,
        )

    def _emit_piece(self, query: str, output: str, piece: PiecePlan) -> str:
        if piece.fallback_text is not None:
            return piece.fallback_text

        grammar = LlamaGrammar.from_string(piece.grammar_text)
        response = self.llm(
            query if query else output,
            grammar=grammar,
            temperature=0,
            max_tokens=piece.max_tokens,
            stop=["<|end|>", "<|start|>"],
        )
        text = response["choices"][0]["text"]
        return text or ""

    def _refresh_state(
        self,
        state: ParserState,
        query: str,
        schema: Dict[str, Any],
        module_fqn: str,
        emitted_fields: List[str],
    ) -> None:
        active_fields, ordered_fields, value_hints = self._compute_plan(
            query=query,
            schema=schema,
            module_fqn=module_fqn,
            emitted_fields=emitted_fields,
        )
        state.active_fields = active_fields
        state.ordered_fields = ordered_fields
        state.value_hints = value_hints
        state.field_index = 0
        state.current = None
        if ordered_fields:
            state.mode = ParserMode.FIELD_SELECT
        else:
            state.mode = ParserMode.DONE

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
        emitted_fields: List[str] = []
        field_attempts: Dict[str, int] = {}
        loosened_fields: set[str] = set()
        backtracks_used = 0

        active_fields, ordered_fields, value_hints = self._compute_plan(
            query=query,
            schema=schema,
            module_fqn=module_fqn,
            emitted_fields=emitted_fields,
        )

        state = ParserState(
            query=query,
            module_fqn=module_fqn,
            schema=schema,
            active_fields=active_fields,
            value_hints=value_hints,
            ordered_fields=ordered_fields,
            mode=ParserMode.HEADER,
        )

        header = self._yaml_header(module_fqn)
        state.output = header
        state.mark_emitted("header")
        state.record_trigger_event("fire:header_emitted")
        state.switch_template(f"module:{module_fqn}")
        state.mode = ParserMode.FIELD_SELECT if ordered_fields else ParserMode.DONE

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

        while budget_left > 0 and not state.done():
            if state.current is None:
                self._refresh_state(state, query, schema, module_fqn, emitted_fields)
                if not state.has_more_fields():
                    state.mode = ParserMode.DONE
                    break

                field_name = state.ordered_fields[state.field_index]
                print("FIELD:", field_name)
                print("EMITTED_FIELDS:", emitted_fields)
                print("ORDERED_FIELDS:", state.ordered_fields)
                print("FIELD_ATTEMPTS:", field_attempts)

                hint = state.value_hints.get(field_name)
                suboptions = dict(state.schema.get("suboptions", {}).get(field_name, {}))

                spec = self._field_spec(schema, field_name)
                kind_str = self._field_kind(spec, hint)

                if kind_str == "list":
                    kind = FieldKind.LIST
                elif kind_str == "dict":
                    kind = FieldKind.DICT
                else:
                    kind = FieldKind.SCALAR

                state.select_current_field(kind=kind, hint=hint, suboptions=suboptions)

            assert state.current is not None
            field_name = state.current.name
            attempt = field_attempts.get(field_name, 0)

            piece = self._piece_for_field(
                state=DecodeState(
                    query=query,
                    module_fqn=module_fqn,
                    schema=schema,
                    active_fields=state.active_fields,
                    value_hints=state.value_hints,
                    emitted_fields=emitted_fields,
                    output=state.output,
                ),
                schema=schema,
                module_fqn=module_fqn,
                field=field_name,
                attempt=attempt,
            )

            if debug:
                print("=" * 100)
                print(f"STATE FIELD: {state.current_field_name()}")
                print(f"ATTEMPT: {attempt}")
                print(f"PIECE: {piece.label}")
                print("CURRENT TEMPLATE:", state.current_template())
                print("GRAMMAR:")
                print(piece.grammar_text)
                print("OUTPUT SO FAR:")
                print(state.output)

            checkpoint = save_checkpoint(
                state=state,
                budget_left=budget_left,
                piece_index=piece_index,
                loosened_fields=loosened_fields,
                backtracks_used=backtracks_used,
                metadata={
                    "piece_label": piece.label,
                    "field": field_name,
                    "attempt": attempt,
                },
            )

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
                    piece_kind=piece.label.split(":")[-1],
                    piece_label=piece.label,
                    generated=generated,
                    debug=debug,
                )

                if any(e.kind == StructuralTriggerKind.INVALID_INDENTATION for e in events):
                    raise RuntimeError(
                        f"invalid indentation for {piece.label}: "
                        f"{[e.reason for e in events if e.kind == StructuralTriggerKind.INVALID_INDENTATION]}"
                    )

                finished_field = field_name
                if finished_field not in emitted_fields:
                    emitted_fields.append(finished_field)
                loosened_fields.discard(finished_field)
                field_attempts.pop(finished_field, None)

                self._refresh_state(state, query, schema, module_fqn, emitted_fields)

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
                    print("=" * 80)
                    print("DEBUG EXCEPTION")
                    print("FIELD:", field_name)
                    print("PIECE:", piece.label)
                    print("ATTEMPT:", attempt)
                    print("EXCEPTION TYPE:", type(exc).__name__)
                    print("EXCEPTION:", exc)
                    traceback.print_exc()
                    print("=" * 80)

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

                field_key = metadata.get("field") or piece.label
                loosened_fields.add(field_key)
                field_attempts[field_key] = field_attempts.get(field_key, 0) + 1
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
                "piece_attempts": field_attempts,
                "generated_fields": emitted_fields,
            }

        return result


StagewiseDynamicDecoderPhase4 = StagewiseDynamicDecoder


def generate_dynamic_yaml(
    query: str,
    schema: Dict[str, Any],
    module_fqn: str,
    max_tokens: int = 256,
    debug: bool = False,
    return_metadata: bool = False,
    enable_backtracking: bool = True,
    max_backtracks: int = 12,
):
    decoder = StagewiseDynamicDecoder()
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