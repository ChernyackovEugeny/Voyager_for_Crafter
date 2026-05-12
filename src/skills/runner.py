"""Compile generated skill source into a callable.

This is the trust boundary for LLM-generated Python skill code. It performs a
small set of deterministic checks, then executes exactly one top-level function
definition in a namespace containing the primitive API.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

from skills import primitives


class SkillLoadError(Exception):
    """Compilation, parsing, or safety-check failure for a skill source."""


@dataclass(frozen=True)
class SkillRuntime:
    """Runtime dependencies exposed to generated skill code."""

    memory: Any | None = None


_FORBIDDEN_NODE_TYPES = (ast.Import, ast.ImportFrom)
_FORBIDDEN_CALL_NAMES = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
})


def _assert_safe(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise SkillLoadError(
                "skill code may not contain imports "
                f"(found {type(node).__name__})"
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SkillLoadError(
                f"skill code may not access dunder attribute '{node.attr}'"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALL_NAMES:
                raise SkillLoadError(
                    f"skill code may not call forbidden builtin '{node.func.id}'"
                )
        if isinstance(node, ast.YieldFrom):
            _assert_valid_yield_from(node)


def _assert_valid_yield_from(node: ast.YieldFrom) -> None:
    value = node.value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return
    name = value.func.id
    sub_generators = {"go_to", "explore_for"}
    if name in primitives.ACTION_PRIMITIVE_NAMES and name not in sub_generators:
        raise SkillLoadError(
            f"skill code may not use 'yield from {name}(...)'; "
            f"{name} returns an int action, not a generator"
        )


def _referenced_names(tree: ast.Module) -> set[str]:
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }


def _assert_single_top_level_function(tree: ast.Module) -> ast.FunctionDef:
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise SkillLoadError(
            "skill must contain exactly one top-level function and no "
            "top-level helper statements"
        )
    return tree.body[0]


def _build_namespace(runtime: SkillRuntime | None) -> dict[str, Any]:
    """Identifiers visible to a skill at runtime."""
    namespace = {
        name: getattr(primitives, name)
        for name in primitives.ACTION_PRIMITIVE_NAMES
    }
    if runtime is not None and runtime.memory is not None:
        namespace.update(_memory_namespace(runtime.memory))
    return namespace


def _memory_namespace(memory: Any) -> dict[str, Any]:
    return {
        "get_memory": memory.get_memory,
        "save_in_memory": memory.memory_add,
        "delete_memory": memory.memory_delete,
        "set_home": memory.set_home,
        "get_home": memory.get_home,
    }


def load_skill(
    source: str,
    runtime: SkillRuntime | None = None,
) -> tuple[str, Callable]:
    """Compile skill source into a callable and return (function_name, func)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SkillLoadError(f"syntax error: {exc}") from exc

    _assert_safe(tree)
    func_def = _assert_single_top_level_function(tree)

    memory_refs = _referenced_names(tree) & primitives.MEMORY_PRIMITIVE_NAMES
    if memory_refs and (runtime is None or runtime.memory is None):
        raise SkillLoadError(
            "skill references memory primitives but no SkillRuntime.memory "
            "was provided"
        )

    func_name = func_def.name
    namespace = _build_namespace(runtime)

    try:
        exec(compile(tree, filename=f"<skill:{func_name}>", mode="exec"), namespace)
    except Exception as exc:
        raise SkillLoadError(f"failed to define skill: {exc}") from exc

    func = namespace.get(func_name)
    if not callable(func):
        raise SkillLoadError(f"top-level def '{func_name}' did not produce a callable")
    return func_name, func
