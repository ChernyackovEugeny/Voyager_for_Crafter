import re

import openai

from config import get_settings
from prompts.codegen_prompt import SYSTEM_PROMPT, format_user_prompt, format_fix_prompt


class CodeGen:
    """Generates and fixes yield-generator skill functions via DeepSeek V3."""

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
    ) -> str:
        """
        Generate a new skill function for the given task.

        Args:
            state_text:       captioner.caption(obs, info) output.
            task:             Task description from the curriculum.
            retrieved_skills: Top-K similar skills from the vector DB.
                              Each dict: {"name", "description", "code"}.

        Returns:
            Python source code of the generated skill (no code fences).
        """
        user_prompt = format_user_prompt(state_text, task, retrieved_skills)
        raw = self._call_api(user_prompt)
        return self._extract_code(raw)

    def fix_bug(
        self,
        skill_code: str,
        error_traceback: str,
        state_text: str,
        task: str,
    ) -> str:
        """
        Ask the LLM to fix a broken skill.

        Args:
            skill_code:       Source code of the broken skill (no fences).
            error_traceback:  traceback.format_exc() output from the executor.
            state_text:       captioner.caption(obs, info) output.
            task:             Original task description.

        Returns:
            Python source code of the fixed skill (no code fences).
        """
        user_prompt = format_fix_prompt(
            state_text, task, skill_code, error_traceback
        )
        raw = self._call_api(user_prompt)
        return self._extract_code(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_api(self, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _extract_code(self, raw_response: str) -> str:
        """
        Strip ```python ... ``` or ``` ... ``` fences from the LLM response.
        Falls back to returning the raw response if no fence is found.
        """
        match = re.search(r"```(?:python)?\n(.*?)```", raw_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return raw_response.strip()
