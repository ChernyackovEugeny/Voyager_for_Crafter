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


_SAFE_BUILTIN_CALLS = frozenset({
    "abs",
    "all",
    "any",
    "bool",
    "callable",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "range",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
})


_CALL_ARG_COUNTS: dict[str, tuple[int, int]] = {
    "noop": (0, 0),
    "move_left": (0, 0),
    "move_right": (0, 0),
    "move_up": (0, 0),
    "move_down": (0, 0),
    "do_action": (0, 0),
    "sleep_action": (0, 0),
    "craft": (1, 1),
    "place": (1, 1),
    "find_nearest": (2, 2),
    "find_nearest_hostile": (1, 1),
    "go_to": (2, 2),
    "get_position": (1, 1),
    "is_hostile_visible": (1, 1),
    "move_away_from_hostile": (1, 1),
    "get_memory": (0, 0),
    "save_in_memory": (2, 2),
    "delete_memory": (1, 1),
    "set_home": (1, 1),
    "get_home": (0, 0),
}


def _assert_safe(
    tree: ast.Module,
    *,
    allowed_skill_names: set[str] | frozenset[str] | None = None,
) -> None:
    top_level_functions = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    allowed_calls = (
        primitives.PRIMITIVE_NAMES
        | _SAFE_BUILTIN_CALLS
        | top_level_functions
        | frozenset(allowed_skill_names or ())
    )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name not in top_level_functions
        ):
            raise SkillLoadError(
                f"skill code may not define nested helper function '{node.name}'"
            )
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
            if node.func.id not in allowed_calls:
                raise SkillLoadError(
                    f"skill code may not call unknown function '{node.func.id}'"
                )
            _assert_call_arity(node)
        if isinstance(node, ast.YieldFrom):
            _assert_valid_yield_from(node)


def _assert_call_arity(node: ast.Call) -> None:
    if not isinstance(node.func, ast.Name):
        return
    name = node.func.id
    if name not in _CALL_ARG_COUNTS:
        return
    if any(isinstance(arg, ast.Starred) for arg in node.args):
        raise SkillLoadError(
            f"skill code may not call primitive '{name}' with starred arguments"
        )
    if node.keywords:
        raise SkillLoadError(
            f"skill code may not call primitive '{name}' with keyword arguments"
        )
    min_args, max_args = _CALL_ARG_COUNTS[name]
    argc = len(node.args)
    if argc < min_args or argc > max_args:
        expected = (
            str(min_args) if min_args == max_args else f"{min_args}-{max_args}"
        )
        raise SkillLoadError(
            f"skill code called primitive '{name}' with {argc} argument(s); "
            f"expected {expected}"
        )


def _assert_valid_yield_from(node: ast.YieldFrom) -> None:
    value = node.value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        return
    name = value.func.id
    sub_generators = {"go_to"}
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
    func_def = tree.body[0]
    _assert_function_contract(func_def)
    _assert_none_safe_function(func_def)
    return func_def


class _NoneSafetyVisitor(ast.NodeVisitor):
    """Reject common generated-code crashes around optional coordinates."""

    _OPTIONAL_COORD_CALLS = frozenset({
        "find_nearest",
        "find_nearest_hostile",
        "get_home",
    })

    def __init__(self) -> None:
        self._maybe_none: set[str] = set()
        self._not_none_stack: list[set[str]] = [set()]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_statements(node.body)

    def _visit_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, ast.Assign):
                self._check_expr(statement.value)
                self._record_assignment(statement)
                continue
            if isinstance(statement, ast.AnnAssign):
                if statement.value is not None:
                    self._check_expr(statement.value)
                    self._record_ann_assignment(statement)
                continue
            if isinstance(statement, ast.AugAssign):
                self._check_expr(statement.value)
                continue
            if isinstance(statement, ast.If):
                self._visit_if(statement)
                continue
            if isinstance(statement, (ast.For, ast.While)):
                self._check_expr(statement.iter if isinstance(statement, ast.For) else statement.test)
                self._visit_statements(statement.body)
                self._visit_statements(statement.orelse)
                continue
            if isinstance(statement, ast.Try):
                self._visit_statements(statement.body)
                for handler in statement.handlers:
                    self._visit_statements(handler.body)
                self._visit_statements(statement.orelse)
                self._visit_statements(statement.finalbody)
                continue
            self._check_expr(statement)

    def _visit_if(self, statement: ast.If) -> None:
        self._check_expr(statement.test)
        body_known, else_known = self._none_guard_scopes(statement.test)

        self._with_not_none(body_known, lambda: self._visit_statements(statement.body))
        self._with_not_none(else_known, lambda: self._visit_statements(statement.orelse))

    def _record_assignment(self, statement: ast.Assign) -> None:
        maybe = self._expr_may_be_none(statement.value)
        for target in statement.targets:
            if isinstance(target, ast.Name):
                self._set_maybe(target.id, maybe)

    def _record_ann_assignment(self, statement: ast.AnnAssign) -> None:
        if isinstance(statement.target, ast.Name):
            self._set_maybe(
                statement.target.id,
                self._expr_may_be_none(statement.value),
            )

    def _set_maybe(self, name: str, maybe: bool) -> None:
        if maybe:
            self._maybe_none.add(name)
        else:
            self._maybe_none.discard(name)

    def _check_expr(self, node: ast.AST | None) -> None:
        if node is None:
            return
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                self._assert_not_optional(child.value, "index")
            elif isinstance(child, ast.Call):
                self._check_call(child)

    def _check_call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id == "go_to" and node.args:
                self._assert_not_optional(node.args[0], "pass to go_to")
            elif node.func.id in {"save_in_memory", "set_home"} and node.args:
                self._assert_not_optional(node.args[-1], f"pass to {node.func.id}")

    def _assert_not_optional(self, expr: ast.AST, action: str) -> None:
        if self._expr_may_be_none(expr):
            name = self._expr_label(expr)
            raise SkillLoadError(
                f"skill may {action} optional value {name}; guard it with "
                "`if value is not None` first"
            )

    def _expr_may_be_none(self, expr: ast.AST | None) -> bool:
        if expr is None:
            return True
        if isinstance(expr, ast.Constant):
            return expr.value is None
        if isinstance(expr, ast.Name):
            return expr.id in self._maybe_none and not self._is_known_not_none(expr.id)
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name):
                return expr.func.id in self._OPTIONAL_COORD_CALLS
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "get":
                if len(expr.args) >= 2:
                    return self._expr_may_be_none(expr.args[1])
                return True
        return False

    def _is_known_not_none(self, name: str) -> bool:
        return any(name in scope for scope in self._not_none_stack)

    def _none_guard(self, expr: ast.AST) -> tuple[str | None, bool]:
        if not isinstance(expr, ast.Compare) or len(expr.ops) != 1:
            return None, False
        if len(expr.comparators) != 1:
            return None, False
        left = expr.left
        right = expr.comparators[0]
        if not isinstance(left, ast.Name):
            return None, False
        if not isinstance(right, ast.Constant) or right.value is not None:
            return None, False
        if isinstance(expr.ops[0], ast.IsNot):
            return left.id, True
        if isinstance(expr.ops[0], ast.Is):
            return left.id, False
        return None, False

    def _none_guard_scopes(self, expr: ast.AST) -> tuple[set[str], set[str]]:
        body_known: set[str] = set()
        else_known: set[str] = set()
        guard_name, guard_is_not_none = self._none_guard(expr)
        if guard_name is not None:
            if guard_is_not_none:
                body_known.add(guard_name)
            else:
                else_known.add(guard_name)
            return body_known, else_known

        if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
            for value in expr.values:
                guard_name, guard_is_not_none = self._none_guard(value)
                if guard_name is not None and guard_is_not_none:
                    body_known.add(guard_name)
        elif isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
            for value in expr.values:
                guard_name, guard_is_not_none = self._none_guard(value)
                if guard_name is not None and not guard_is_not_none:
                    else_known.add(guard_name)
        return body_known, else_known

    def _with_not_none(self, names: set[str], callback) -> None:
        self._not_none_stack.append(names)
        try:
            callback()
        finally:
            self._not_none_stack.pop()

    @staticmethod
    def _expr_label(expr: ast.AST) -> str:
        if isinstance(expr, ast.Name):
            return repr(expr.id)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            return f"{expr.func.id}(...)"
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            return f".{expr.func.attr}(...)"
        return ast.dump(expr, include_attributes=False)


def _assert_none_safe_function(func_def: ast.FunctionDef) -> None:
    _NoneSafetyVisitor().visit(func_def)


def _assert_function_contract(func_def: ast.FunctionDef) -> None:
    args = func_def.args
    if (
        len(args.args) != 1
        or args.args[0].arg != "state"
        or args.posonlyargs
        or args.vararg is not None
        or args.kwonlyargs
        or args.kwarg is not None
        or args.defaults
        or args.kw_defaults
    ):
        raise SkillLoadError(
            f"skill function '{func_def.name}' must accept exactly one "
            "argument named 'state'"
        )
    if not any(
        isinstance(node, (ast.Yield, ast.YieldFrom))
        for node in ast.walk(func_def)
    ):
        raise SkillLoadError(
            f"skill function '{func_def.name}' must be a generator with at "
            "least one yield"
        )


def _build_namespace(runtime: SkillRuntime | None) -> dict[str, Any]:
    """Identifiers visible to a skill at runtime."""
    namespace = {
        name: getattr(primitives, name)
        for name in primitives.ACTION_PRIMITIVE_NAMES
    }
    if runtime is not None and runtime.memory is not None:
        namespace.update(_memory_namespace(runtime.memory))
    return namespace


def _wrap_skill_returns_state(func: Callable) -> Callable:
    """Make old saved skills safe when called with `yield from`.

    Historic skills often use bare `return`, which makes `yield from skill(state)`
    evaluate to None. The wrapper preserves normal yielded actions while
    replacing a missing StopIteration.value with the latest state sent by the
    executor.
    """

    def wrapped(state):
        latest_state = state
        gen = func(state)
        try:
            action = next(gen)
            while True:
                try:
                    sent_state = yield action
                except GeneratorExit:
                    raise
                if sent_state is not None:
                    latest_state = sent_state
                try:
                    action = gen.send(sent_state)
                except StopIteration as stop:
                    return stop.value if stop.value is not None else latest_state
        except StopIteration as stop:
            return stop.value if stop.value is not None else latest_state

    wrapped.__name__ = getattr(func, "__name__", "wrapped_skill")
    wrapped.__doc__ = getattr(func, "__doc__", None)
    return wrapped


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
    extra_skills: list[tuple[str, str]] | None = None,
    allowed_skill_names: set[str] | frozenset[str] | None = None,
) -> tuple[str, Callable]:
    """Compile skill source into a callable and return (function_name, func).

    extra_skills: optional list of (skill_name, skill_code) pairs to compile
    into the namespace BEFORE the main skill, so the main skill can call them
    by name. Individual extra_skills that fail safety/parse checks are silently
    skipped — they remain available in the prompt as inspiration but won't
    crash the main load.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SkillLoadError(f"syntax error: {exc}") from exc

    extra_names = {name for name, _ in extra_skills or []}
    explicit_names = set(allowed_skill_names or ())
    _assert_safe(tree, allowed_skill_names=extra_names | explicit_names)
    func_def = _assert_single_top_level_function(tree)
    required_extra_names = _referenced_names(tree) & extra_names

    memory_refs = _referenced_names(tree) & primitives.MEMORY_PRIMITIVE_NAMES
    if memory_refs and (runtime is None or runtime.memory is None):
        raise SkillLoadError(
            "skill references memory primitives but no SkillRuntime.memory "
            "was provided"
        )

    func_name = func_def.name
    namespace = _build_namespace(runtime)

    if extra_skills:
        loaded_extra_names: set[str] = set()
        failed_extra_errors: dict[str, str] = {}
        for extra_name, extra_code in extra_skills:
            if extra_name == func_name:
                continue
            try:
                extra_tree = ast.parse(extra_code)
                _assert_safe(extra_tree, allowed_skill_names=extra_names)
                extra_func_def = _assert_single_top_level_function(extra_tree)
                memory_refs = (
                    _referenced_names(extra_tree) & primitives.MEMORY_PRIMITIVE_NAMES
                )
                if memory_refs and (runtime is None or runtime.memory is None):
                    raise SkillLoadError(
                        "extra skill references memory primitives but no "
                        "SkillRuntime.memory was provided"
                    )
                exec(
                    compile(
                        extra_tree,
                        filename=f"<extra_skill:{extra_name}>",
                        mode="exec",
                    ),
                    namespace,
                )
                extra_func = namespace.get(extra_func_def.name)
                if callable(extra_func):
                    wrapped_extra = _wrap_skill_returns_state(extra_func)
                    namespace[extra_func_def.name] = wrapped_extra
                    namespace[extra_name] = wrapped_extra
                    loaded_extra_names.add(extra_name)
            except (SkillLoadError, SyntaxError, Exception) as exc:
                failed_extra_errors[extra_name] = str(exc)
                continue
        missing_required = required_extra_names - loaded_extra_names
        if missing_required:
            details = "; ".join(
                f"{name}: {failed_extra_errors.get(name, 'not loaded')}"
                for name in sorted(missing_required)
            )
            raise SkillLoadError(f"required extra skill failed to load: {details}")

    try:
        exec(compile(tree, filename=f"<skill:{func_name}>", mode="exec"), namespace)
    except Exception as exc:
        raise SkillLoadError(f"failed to define skill: {exc}") from exc

    func = namespace.get(func_name)
    if not callable(func):
        raise SkillLoadError(f"top-level def '{func_name}' did not produce a callable")
    return func_name, _wrap_skill_returns_state(func)
