from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class DecodeCheckpoint:
    state: Any
    budget_left: int
    piece_index: int
    loosened_fields: set[str] = field(default_factory=set)
    backtracks_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def save_checkpoint(
    state: Any,
    budget_left: int,
    piece_index: int,
    loosened_fields: set[str],
    backtracks_used: int,
    metadata: Dict[str, Any] | None = None,
) -> DecodeCheckpoint:
    return DecodeCheckpoint(
        state=deepcopy(state),
        budget_left=int(budget_left),
        piece_index=int(piece_index),
        loosened_fields=set(loosened_fields),
        backtracks_used=int(backtracks_used),
        metadata=dict(metadata or {}),
    )


def restore_checkpoint(
    checkpoint: DecodeCheckpoint,
) -> Tuple[Any, int, int, set[str], int, Dict[str, Any]]:
    return (
        deepcopy(checkpoint.state),
        int(checkpoint.budget_left),
        int(checkpoint.piece_index),
        set(checkpoint.loosened_fields),
        int(checkpoint.backtracks_used),
        dict(checkpoint.metadata),
    )