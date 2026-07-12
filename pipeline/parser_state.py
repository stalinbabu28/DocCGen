from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class ParserMode(Enum):
    HEADER = auto()
    FIELD_SELECT = auto()
    LIST_ITEMS = auto()
    DICT_ITEMS = auto()
    DONE = auto()


class FieldKind(Enum):
    SCALAR = "scalar"
    LIST = "list"
    DICT = "dict"


@dataclass
class FieldCursor:
    name: str
    kind: FieldKind
    hint: Any = None
    suboptions: Dict[str, Any] = dataclass_field(default_factory=dict)
    started: bool = False
    item_index: int = 0
    pair_index: int = 0
    items: List[Any] = dataclass_field(default_factory=list)
    pairs: List[Tuple[str, Any]] = dataclass_field(default_factory=list)


@dataclass
class ParserState:
    query: str
    module_fqn: str
    schema: Dict[str, Any]
    active_fields: List[str]
    value_hints: Dict[str, Any]
    ordered_fields: List[str]
    mode: ParserMode = ParserMode.HEADER
    field_index: int = 0
    current: Optional[FieldCursor] = None
    output: str = ""

    parser_log: List[str] = dataclass_field(default_factory=list)
    decoder_log: List[str] = dataclass_field(default_factory=list)
    trigger_log: List[str] = dataclass_field(default_factory=list)

    template_stack: List[str] = dataclass_field(default_factory=lambda: ["root"])
    current_indent: int = 0

    def has_more_fields(self) -> bool:
        return self.field_index < len(self.ordered_fields)

    def current_field_name(self) -> Optional[str]:
        return self.current.name if self.current else None

    def current_template(self) -> str:
        return self.template_stack[-1] if self.template_stack else "root"

    def record_parser_event(self, label: str) -> None:
        self.parser_log.append(label)

    def record_decoder_event(self, label: str) -> None:
        self.decoder_log.append(label)

    def record_trigger_event(self, label: str) -> None:
        self.trigger_log.append(label)

    @property
    def transition_log(self) -> List[str]:
        return self.parser_log + self.decoder_log + self.trigger_log

    @property
    def structural_events(self) -> List[str]:
        return self.trigger_log

    def push_template(self, label: str) -> None:
        self.template_stack.append(label)
        self.record_parser_event(f"push:{label}")

    def switch_template(self, label: str) -> None:
        if self.template_stack:
            self.template_stack[-1] = label
        else:
            self.template_stack.append(label)
        self.record_parser_event(f"switch:{label}")

    def pop_template(self) -> None:
        if len(self.template_stack) > 1:
            label = self.template_stack.pop()
            self.record_parser_event(f"pop:{label}")

    def select_current_field(
        self,
        kind: FieldKind,
        hint: Any,
        suboptions: Dict[str, Any],
    ) -> bool:
        if not self.has_more_fields():
            self.mode = ParserMode.DONE
            return False

        name = self.ordered_fields[self.field_index]
        items = list(hint) if isinstance(hint, list) else []
        pairs = list(hint.items()) if isinstance(hint, dict) else []

        self.current = FieldCursor(
            name=name,
            kind=kind,
            hint=hint,
            suboptions=suboptions,
            items=items,
            pairs=pairs,
        )
        self.record_parser_event(f"select:{name}:{kind.value}")
        self.push_template(f"field:{name}:{kind.value}")
        self.mode = ParserMode.FIELD_SELECT
        return True

    def finish_current_field(self) -> None:
        if self.current is not None:
            self.record_parser_event(f"finish:{self.current.name}")

        self.current = None
        self.field_index += 1
        self.pop_template()
        self.mode = ParserMode.FIELD_SELECT if self.has_more_fields() else ParserMode.DONE

    def done(self) -> bool:
        return self.mode == ParserMode.DONE

    def mark_emitted(self, label: str) -> None:
        self.record_decoder_event(f"emit:{label}")

    # Backward-compatible aliases.
    def record_structural_event(self, label: str) -> None:
        self.record_trigger_event(label)