"""Deterministic skill description generation."""
from __future__ import annotations

import ast
import logging

from skills.primitives import PRIMITIVE_NAMES

logger = logging.getLogger(__name__)


def extract_primitives_called(code: str) -> list[str]:
    """Return sorted primitive function names called by the skill code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        logger.warning("extract_primitives_called: syntax error (%s)", exc)
        return []

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in PRIMITIVE_NAMES:
                found.add(node.func.id)
    return sorted(found)


def compose_description(name: str, task: str, code: str) -> str:
    """Build the text used for embedding and retrieval.

    The skill name is intentionally ignored: generated names tend to be noisy,
    while the task and primitive calls carry the useful retrieval signal.
    """
    del name
    primitives = extract_primitives_called(code)
    if not primitives:
        return task
    return f"{task}. Uses: {', '.join(primitives)}"
