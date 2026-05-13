"""Prompt templates for compile/load-time skill repair."""
from __future__ import annotations

FIX_BUG_TEMPLATE_ID = "fix_bug.v1"

FIX_BUG_SYSTEM_PROMPT = """\
You repair Python generator functions for a Crafter game agent.

Your task is narrow: make one broken skill load successfully under the skill
runtime. Do not redesign the strategy. Do not add new goals. Do not optimize
behavior. Only fix technical errors reported by validation.

Skill contract:
- Output exactly one top-level Python function.
- The function must be a generator that yields integer actions.
- Use `yield from` only with generator primitives such as `go_to(...)` and
  `explore_for(...)`.
- Do not use `yield from` with action primitives such as `move_left(...)`,
  `move_right(...)`, `move_up(...)`, `move_down(...)`, `do_action(...)`,
  `sleep(...)`, `place(...)`, or `make(...)`; those return one integer action.
- Do not import modules.
- Do not call forbidden builtins such as eval, exec, compile, open, getattr,
  setattr, globals, locals, vars, dir, or __import__.
- Do not create helper functions or top-level statements.
- Preserve the original function name and purpose whenever possible.

Return only the corrected function in a ```python ... ``` fenced block.
"""


def format_fix_bug_prompt(
    *,
    state_text: str,
    task: str,
    skill_code: str,
    error_traceback: str,
) -> str:
    """Build the user message for a compile/load-time fix_bug call."""
    return (
        f"## Task\n{task}\n\n"
        f"## Current Game State\n{state_text}\n\n"
        f"## Broken Skill\n"
        f"```python\n{skill_code}\n```\n\n"
        f"## Validation Error\n"
        f"```\n{error_traceback}\n```\n\n"
        f"## Your Job\n"
        f"Fix only the technical issue that prevents this skill from loading.\n"
        f"Keep the same task intent and avoid behavioral rewrites.\n"
        f"Return only one corrected Python generator function."
    )
