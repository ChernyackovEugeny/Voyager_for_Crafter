from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

import openai

from config import get_settings
from llm.pricing import compute_cost
from prompts.reflection_prompt import (
    REFLECTION_TEMPLATE_ID,
    SYSTEM_PROMPT,
    format_user_prompt,
)


@dataclass(frozen=True)
class FailureContext:
    """Inputs needed to improve a previously saved skill after reuse failure."""

    task_description: str
    failure_reason: str
    skill_code: str
    state_snapshot: dict[str, Any]
    error_traceback: str | None = None


@dataclass(frozen=True)
class ReflectionCall:
    """Reflection result plus billing/latency metadata."""

    code: str
    raw_response: str
    model: str
    prompt_template_id: str
    prompt_hash: str
    prompt_tokens: int
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    latency_ms: int
    cost_usd: float
    prompt_text: str | None = None

    @property
    def tokens_in(self) -> int:
        return self.prompt_tokens

    @property
    def tokens_out(self) -> int:
        return self.completion_tokens


class Reflection:
    """Improves saved skills after behavioral failures using DeepSeek R1."""

    def __init__(
        self,
        *,
        client=None,
        model: str | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        cfg = get_settings().llm
        self._model = model or cfg.reflection_model
        self._temperature = (
            cfg.reflection_temperature if temperature is None else temperature
        )
        self._client = client or openai.OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            timeout=timeout_s or cfg.reflection_timeout_s,
        )

    def improve_skill(self, ctx: FailureContext) -> ReflectionCall:
        user_prompt = format_user_prompt(
            task_description=ctx.task_description,
            failure_reason=ctx.failure_reason,
            skill_code=ctx.skill_code,
            state_snapshot=ctx.state_snapshot,
            error_traceback=ctx.error_traceback,
        )
        prompt_hash = self._prompt_hash(user_prompt)
        started = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        raw = response.choices[0].message.content
        usage = self._extract_usage(response)
        cost = compute_cost(
            self._model,
            prompt_cache_hit_tokens=usage["prompt_cache_hit_tokens"],
            prompt_cache_miss_tokens=usage["prompt_cache_miss_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
        return ReflectionCall(
            code=self._extract_code(raw),
            raw_response=raw,
            model=self._model,
            prompt_template_id=REFLECTION_TEMPLATE_ID,
            prompt_hash=prompt_hash,
            prompt_tokens=usage["prompt_tokens"],
            prompt_cache_hit_tokens=usage["prompt_cache_hit_tokens"],
            prompt_cache_miss_tokens=usage["prompt_cache_miss_tokens"],
            completion_tokens=usage["completion_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            latency_ms=latency_ms,
            cost_usd=cost,
            prompt_text=user_prompt,
        )

    @staticmethod
    def _prompt_hash(user_prompt: str) -> str:
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        return hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _extract_usage(response) -> dict[str, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {
                "prompt_tokens": 0,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": None,
            }

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        hit_tokens = getattr(usage, "prompt_cache_hit_tokens", None)
        miss_tokens = getattr(usage, "prompt_cache_miss_tokens", None)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        if hit_tokens is None and miss_tokens is None:
            hit_tokens = 0
            miss_tokens = prompt_tokens
        else:
            hit_tokens = int(hit_tokens or 0)
            miss_tokens = int(miss_tokens or 0)
            if prompt_tokens == 0:
                prompt_tokens = hit_tokens + miss_tokens

        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = (
            getattr(details, "reasoning_tokens", None) if details else None
        )
        if reasoning_tokens is not None:
            reasoning_tokens = int(reasoning_tokens)

        return {
            "prompt_tokens": prompt_tokens,
            "prompt_cache_hit_tokens": int(hit_tokens),
            "prompt_cache_miss_tokens": int(miss_tokens),
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    @staticmethod
    def _extract_code(raw_response: str) -> str:
        match = re.search(r"```(?:python)?\n(.*?)```", raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_response.strip()
