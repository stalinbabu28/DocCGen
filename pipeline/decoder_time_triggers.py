from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from pipeline.parser_state import ParserMode, ParserState


class StructuralTriggerKind(Enum):
    HEADER_EMITTED = auto()
    SCALAR_EMITTED = auto()
    LIST_HEADER_EMITTED = auto()
    LIST_ITEM_EMITTED = auto()
    DICT_HEADER_EMITTED = auto()
    DICT_PAIR_EMITTED = auto()
    BLOCK_EMITTED = auto()
    INVALID_INDENTATION = auto()


@dataclass
class StructuralTriggerEvent:
    kind: StructuralTriggerKind
    field: Optional[str] = None
    piece_label: Optional[str] = None
    indent: int = 0
    expected_indent: int = 0
    raw_text: str = ""
    reason: str = ""


class DecoderTimeTriggerEngine:
    """
    Phase 3 structural trigger engine.

    Watches emitted YAML structure and mutates parser state after each piece:
    - header emission
    - scalar field emission
    - list header -> list item(s)
    - dict header -> dict pair(s)
    - invalid indentation reporting

    Phase 4 will add recovery/backtracking for invalid indentation.
    """

    def _first_nonempty_line(self, text: str) -> str:
        for line in text.splitlines():
            if line.strip():
                return line
        return ""

    def _leading_spaces(self, line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _expected_indent(self, piece_kind: str) -> int:
        if piece_kind in {"scalar", "scalar-direct", "list-header", "dict-header"}:
            return 4
        if piece_kind in {"list-item", "dict-pair"}:
            return 8
        if piece_kind == "header":
            return 0
        return -1

    def _classify_piece_kind(self, piece_kind: str) -> StructuralTriggerKind:
        mapping = {
            "header": StructuralTriggerKind.HEADER_EMITTED,
            "scalar": StructuralTriggerKind.SCALAR_EMITTED,
            "scalar-direct": StructuralTriggerKind.SCALAR_EMITTED,
            "list-header": StructuralTriggerKind.LIST_HEADER_EMITTED,
            "list-item": StructuralTriggerKind.LIST_ITEM_EMITTED,
            "dict-header": StructuralTriggerKind.DICT_HEADER_EMITTED,
            "dict-pair": StructuralTriggerKind.DICT_PAIR_EMITTED,
            "list-block": StructuralTriggerKind.BLOCK_EMITTED,
            "dict-block": StructuralTriggerKind.BLOCK_EMITTED,
        }
        return mapping.get(piece_kind, StructuralTriggerKind.BLOCK_EMITTED)

    def _log_event(self, state: ParserState, event: StructuralTriggerEvent) -> None:
        label = f"fire:{event.kind.name.lower()}"
        if event.field:
            label = f"{label}:{event.field}"
        if event.reason:
            label = f"{label}:{event.reason}"
        state.record_trigger_event(label)

    def classify(
        self,
        state: ParserState,
        piece_kind: str,
        piece_label: str,
        generated: str,
    ) -> StructuralTriggerEvent:
        expected_indent = self._expected_indent(piece_kind)
        first_line = self._first_nonempty_line(generated)
        actual_indent = self._leading_spaces(first_line) if first_line else 0

        field = state.current.name if state.current else None
        event = StructuralTriggerEvent(
            kind=self._classify_piece_kind(piece_kind),
            field=field,
            piece_label=piece_label,
            indent=actual_indent,
            expected_indent=max(expected_indent, 0),
            raw_text=generated,
            reason="",
        )

        if expected_indent >= 0 and first_line and actual_indent != expected_indent:
            event.kind = StructuralTriggerKind.INVALID_INDENTATION
            event.reason = f"expected={expected_indent},actual={actual_indent}"
        return event

    def apply(
        self,
        state: ParserState,
        piece_kind: str,
        piece_label: str,
        generated: str,
        debug: bool = False,
    ) -> List[StructuralTriggerEvent]:
        events: List[StructuralTriggerEvent] = []
        event = self.classify(state, piece_kind, piece_label, generated)
        events.append(event)

        if debug:
            state.record_trigger_event(
                f"debug:trigger:{event.kind.name.lower()}:{event.field or 'none'}"
            )

        self._log_event(state, event)

        if event.kind == StructuralTriggerKind.INVALID_INDENTATION:
            return events

        ctx = state.current

        if event.kind == StructuralTriggerKind.HEADER_EMITTED:
            state.switch_template(f"module:{state.module_fqn}")
            state.mode = ParserMode.FIELD_SELECT
            return events

        if ctx is None:
            return events

        if event.kind == StructuralTriggerKind.SCALAR_EMITTED:
            state.current_indent = 4
            state.finish_current_field()
            return events

        if event.kind == StructuralTriggerKind.LIST_HEADER_EMITTED:
            ctx.started = True
            state.current_indent = 4
            state.switch_template(f"list:{ctx.name}")
            state.mode = ParserMode.LIST_ITEMS
            return events

        if event.kind == StructuralTriggerKind.LIST_ITEM_EMITTED:
            ctx.item_index += 1
            state.current_indent = 8
            if isinstance(ctx.hint, list) and ctx.item_index >= len(ctx.hint):
                state.record_trigger_event(f"list-complete:{ctx.name}")
                state.finish_current_field()
            else:
                state.mode = ParserMode.LIST_ITEMS
            return events

        if event.kind == StructuralTriggerKind.DICT_HEADER_EMITTED:
            ctx.started = True
            state.current_indent = 4
            state.switch_template(f"dict:{ctx.name}")
            state.mode = ParserMode.DICT_ITEMS
            return events

        if event.kind == StructuralTriggerKind.DICT_PAIR_EMITTED:
            ctx.pair_index += 1
            state.current_indent = 8
            if isinstance(ctx.hint, dict) and ctx.pair_index >= len(ctx.hint):
                state.record_trigger_event(f"dict-complete:{ctx.name}")
                state.finish_current_field()
            else:
                state.mode = ParserMode.DICT_ITEMS
            return events

        if event.kind == StructuralTriggerKind.BLOCK_EMITTED:
            state.finish_current_field()
            return events

        return events