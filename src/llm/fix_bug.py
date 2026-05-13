from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

import openai

from config import get_settings
from llm.pricing import compute_cost
from prompts.fix_bug_prompt import (
    FIX_BUG_SYSTEM_PROMPT,
    FIX_BUG_TEMPLATE_ID,
    format_fix_bug_prompt,
)


@dataclass(frozen=True)
class FixBugCall:
    """Compile/load-time repair result plus billing/latency metadata."""

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

    @property
    def tokens_in(self) -> int:
        return self.prompt_tokens

    @property
    def tokens_out(self) -> int:
        return self.completion_tokens


class FixBugError(Exception):
    """fix_bug LLM call failed after prompt metadata was known."""

    def __init__(
        self,
        message: str,
        *,
        prompt_template_id: str,
        prompt_hash: str,
        latency_ms: int,
    ) -> None:
        super().__init__(message)
        self.prompt_template_id = prompt_template_id
        self.prompt_hash = prompt_hash
        self.latency_ms = latency_ms


class FixBug:
    """Repairs generated skills that fail deterministic load validation."""

    def __init__(
        self,
        *,
        client=None,
        model: str | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        cfg = get_settings().llm
        self._model = model or cfg.codegen_model
        self._temperature = cfg.codegen_temperature if temperature is None else temperature
        self._client = client or openai.OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            timeout=timeout_s or cfg.request_timeout_s,
        )

    def fix(
        self,
        *,
        skill_code: str,
        error_traceback: str,
        state_text: str,
        task: str,
    ) -> FixBugCall:
        """Return a technically repaired version of a broken generated skill."""
        user_prompt = format_fix_bug_prompt(
            state_text=state_text,
            task=task,
            skill_code=skill_code,
            error_traceback=error_traceback,
        )
        prompt_hash = self._prompt_hash(user_prompt)
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": FIX_BUG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            raise FixBugError(
                str(exc),
                prompt_template_id=FIX_BUG_TEMPLATE_ID,
                prompt_hash=prompt_hash,
                latency_ms=latency_ms,
            ) from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        raw = response.choices[0].message.content
        usage = self._extract_usage(response)
        cost = compute_cost(
            self._model,
            prompt_cache_hit_tokens=usage["prompt_cache_hit_tokens"],
            prompt_cache_miss_tokens=usage["prompt_cache_miss_tokens"],
            completion_tokens=usage["completion_tokens"],
        )
        return FixBugCall(
            code=self._extract_code(raw),
            raw_response=raw,
            model=self._model,
            prompt_template_id=FIX_BUG_TEMPLATE_ID,
            prompt_hash=prompt_hash,
            prompt_tokens=usage["prompt_tokens"],
            prompt_cache_hit_tokens=usage["prompt_cache_hit_tokens"],
            prompt_cache_miss_tokens=usage["prompt_cache_miss_tokens"],
            completion_tokens=usage["completion_tokens"],
            reasoning_tokens=usage["reasoning_tokens"],
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    @staticmethod
    def _prompt_hash(user_prompt: str) -> str:
        full_prompt = f"{FIX_BUG_SYSTEM_PROMPT}\n\n{user_prompt}"
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
