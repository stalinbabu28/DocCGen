from __future__ import annotations

from typing import Any, Dict

from pipeline.dynamic_decoder import StagewiseDynamicDecoder, generate_dynamic_yaml

Phase3ParserStateDecoder = StagewiseDynamicDecoder
Phase2ParserStateDecoder = StagewiseDynamicDecoder


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
    return generate_dynamic_yaml(
        query=query,
        schema=schema,
        module_fqn=module_fqn,
        max_tokens=max_tokens,
        debug=debug,
        return_metadata=return_metadata,
        enable_backtracking=enable_backtracking,
        max_backtracks=max_backtracks,
    )