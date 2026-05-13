from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

import openai

from config import get_settings
from llm.pricing import compute_cost
from prompts.codegen_prompt import (
    CODEGEN_TEMPLATE_ID,
    SYSTEM_PROMPT,
    format_user_prompt,
)


@dataclass(frozen=True)
class CodeGenCall:
    """Code generation result plus billing/latency metadata."""

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


class CodeGenError(Exception):
    """LLM call failed after prompt metadata was known."""

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


class CodeGen:
    """Generates yield-generator skill functions via DeepSeek V3."""

    def __init__(self):
        cfg = get_settings().llm
        self._model = cfg.codegen_model
        self._temperature = cfg.codegen_temperature
        self._client = openai.OpenAI(
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
            timeout=cfg.request_timeout_s,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_code(
        self,
        state_text: str,
        task: str,
        retrieved_skills: list[dict],
        previous_failure: tuple[str, str] | None = None,
    ) -> CodeGenCall:
        """
        Generate a new skill function for the given task.

        Args:
            state_text:       captioner.caption(obs, info) output.
            task:             Task description from the curriculum.
            retrieved_skills: Top-K similar skills from the vector DB.
                              Each dict: {"name", "description", "code"}.
            previous_failure: Optional (broken_code, failure_reason) from the
                              previous codegen attempt for this same task.

        Returns:
            CodeGenCall with source code and API usage metadata.
        """
        user_prompt = format_user_prompt(
            state_text, task, retrieved_skills, previous_failure=previous_failure
        )
        return self._call_api(user_prompt, CODEGEN_TEMPLATE_ID)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, user_prompt: str, template_id: str) -> CodeGenCall:
        prompt_hash = self._prompt_hash(user_prompt)
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            raise CodeGenError(
                str(exc),
                prompt_template_id=template_id,
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
        return CodeGenCall(
            code=self._extract_code(raw),
            raw_response=raw,
            model=self._model,
            prompt_template_id=template_id,
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

    def _extract_code(self, raw_response: str) -> str:
        """
        Strip ```python ... ``` or ``` ... ``` fences from the LLM response.
        Falls back to returning the raw response if no fence is found.
        """
        match = re.search(r"```(?:python)?\n(.*?)```", raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_response.strip()
