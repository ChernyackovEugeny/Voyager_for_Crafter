import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.reflection import FailureContext, Reflection


class StubCompletions:
    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=100,
            prompt_cache_hit_tokens=30,
            prompt_cache_miss_tokens=70,
            completion_tokens=20,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
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


class ReflectionTests(unittest.TestCase):
    def test_improve_skill_returns_extracted_code_and_metadata(self):
        code = "def collect_wood(state):\n    state = yield 1\n"
        client = StubClient(f"```python\n{code}\n```")
        reflection = Reflection(
            client=client,
            model="deepseek-reasoner",
            temperature=0.1,
        )

        call = reflection.improve_skill(
            FailureContext(
                task_description="collect wood",
                failure_reason="timeout",
                skill_code="def collect_wood(state):\n    state = yield 0\n",
                state_snapshot={"info": {"inventory": {"health": 9}}},
            )
        )

        self.assertEqual(call.code, code.strip())
        self.assertEqual(call.model, "deepseek-reasoner")
        self.assertEqual(call.prompt_template_id, "reflection.v1")
        self.assertEqual(call.prompt_tokens, 100)
        self.assertEqual(call.reasoning_tokens, 10)
        self.assertEqual(client.completions.calls[0]["temperature"], 0.1)

    def test_extract_code_falls_back_to_raw_response(self):
        client = StubClient("def f(state):\n    yield 0")
        reflection = Reflection(client=client, model="deepseek-reasoner")

        call = reflection.improve_skill(
            FailureContext(
                task_description="noop",
                failure_reason="task_incomplete",
                skill_code="def f(state):\n    yield 0",
                state_snapshot={},
            )
        )

        self.assertEqual(call.code, "def f(state):\n    yield 0")


if __name__ == "__main__":
    unittest.main()
