import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.fix_bug import FixBug
from prompts.fix_bug_prompt import FIX_BUG_SYSTEM_PROMPT


class StubCompletions:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=90,
            prompt_cache_hit_tokens=20,
            prompt_cache_miss_tokens=70,
            completion_tokens=15,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=None),
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.response_text),
                )
            ],
            usage=usage,
        )


class StubClient:
    def __init__(self, response_text):
        self.completions = StubCompletions(response_text)
        self.chat = SimpleNamespace(completions=self.completions)


class FixBugTests(unittest.TestCase):
    def test_fix_returns_extracted_code_and_uses_dedicated_prompt(self):
        code = "def collect_wood(state):\n    state = yield 1\n"
        client = StubClient(f"```python\n{code}\n```")
        fixer = FixBug(client=client, model="deepseek-chat", temperature=0.0)

        call = fixer.fix(
            skill_code="def collect_wood(state):\n    state = yield from move_left(state)",
            error_traceback="yield from move_left is invalid",
            state_text="Inventory: none",
            task="Chop a tree to obtain wood.",
        )

        request = client.completions.calls[0]
        self.assertEqual(call.code, code.strip())
        self.assertEqual(call.prompt_template_id, "fix_bug.v1")
        self.assertEqual(call.prompt_tokens, 90)
        self.assertEqual(request["model"], "deepseek-chat")
        self.assertEqual(request["temperature"], 0.0)
        self.assertIs(request["messages"][0]["content"], FIX_BUG_SYSTEM_PROMPT)
        self.assertIn(
            "Validation Or Runtime Error",
            request["messages"][1]["content"],
        )

    def test_extract_code_falls_back_to_raw_response(self):
        client = StubClient("def f(state):\n    yield 0")
        fixer = FixBug(client=client, model="deepseek-chat")

        call = fixer.fix(
            skill_code="def f(state):\n    yield from move_left(state)",
            error_traceback="invalid yield from",
            state_text="",
            task="noop",
        )

        self.assertEqual(call.code, "def f(state):\n    yield 0")


if __name__ == "__main__":
    unittest.main()
