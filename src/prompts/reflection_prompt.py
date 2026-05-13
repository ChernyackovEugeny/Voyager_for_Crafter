"""Prompt templates for R1-based skill reflection.

The system prompt is intentionally static so provider-side prompt caching can
reuse the large Crafter/primitive instructions across reflection calls.
"""
from __future__ import annotations

import json

from prompts.codegen_prompt import SYSTEM_PROMPT as CODEGEN_SYSTEM_PROMPT


REFLECTION_TEMPLATE_ID = "reflection.v1"

SYSTEM_PROMPT = f"""{CODEGEN_SYSTEM_PROMPT}

================================================================================
## Reflection Mode
================================================================================

You are now reviewing an existing saved skill that failed when reused.
Analyze the failure privately and return only an improved Python generator
function.

Rules:
  - Keep the same function name and signature whenever possible.
  - Preserve the original task purpose.
  - Prefer small, robust changes over broad rewrites.
  - Do not add explanations outside the code fence.
  - Return exactly one corrected Python function in a ```python code fence.
"""


def format_user_prompt(
    *,
    task_description: str,
    failure_reason: str,
    skill_code: str,
    state_snapshot: dict,
    error_traceback: str | None = None,
) -> str:
    snapshot_json = json.dumps(state_snapshot, ensure_ascii=False, indent=2, default=str)
    return f"""\
## Original Task
{task_description}

## Skill Code That Failed
```python
{skill_code}
```

## Failure Reason
{failure_reason}

## State at Failure
```json
{snapshot_json}
```

## Error Traceback
{error_traceback or "N/A"}

## Your Job
Identify why this saved skill failed in this state and produce an improved
version. Return only the improved Python function in a ```python code fence.
"""
