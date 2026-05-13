import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.codegen import CodeGen, CodeGenError
from llm.pricing import compute_cost


class _Usage:
    prompt_tokens = 100
    prompt_cache_hit_tokens = 30
    prompt_cache_miss_tokens = 70
    completion_tokens = 20


class _Message:
    content = "```python\ndef f(state):\n    state = yield 0\n```"


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]
    usage = _Usage()


class _Completions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _Chat:
    def __init__(self, completions):
        self.completions = completions


class _Client:
    def __init__(self, completions):
        self.chat = _Chat(completions)


class CodegenAnalyticsTests(unittest.TestCase):
    def test_compute_cost_uses_cache_hit_and_miss_prices(self):
        cost = compute_cost(
            "deepseek-v4-flash",
            prompt_cache_hit_tokens=100_000,
            prompt_cache_miss_tokens=100_000,
            completion_tokens=100_000,
        )

        self.assertAlmostEqual(cost, 0.00028 + 0.014 + 0.028)

    def test_call_api_returns_codegen_call_with_usage_and_cost(self):
        completions = _Completions(response=_Response())
        codegen = CodeGen.__new__(CodeGen)
        codegen._model = "deepseek-v4-flash"
        codegen._temperature = 0.0
        codegen._client = _Client(completions)

        call = codegen._call_api("hello", "codegen.v1")

        self.assertIn("def f", call.code)
        self.assertEqual(call.prompt_tokens, 100)
        self.assertEqual(call.prompt_cache_hit_tokens, 30)
        self.assertEqual(call.prompt_cache_miss_tokens, 70)
        self.assertEqual(call.completion_tokens, 20)
        self.assertGreaterEqual(call.latency_ms, 0)
        self.assertGreater(call.cost_usd, 0)
        self.assertEqual(call.tokens_in, 100)
        self.assertEqual(call.tokens_out, 20)

    def test_call_api_wraps_errors_with_prompt_metadata(self):
        completions = _Completions(error=RuntimeError("network down"))
        codegen = CodeGen.__new__(CodeGen)
        codegen._model = "deepseek-v4-flash"
        codegen._temperature = 0.0
        codegen._client = _Client(completions)

        with self.assertRaises(CodeGenError) as ctx:
            codegen._call_api("hello", "codegen.v1")

        self.assertEqual(ctx.exception.prompt_template_id, "codegen.v1")
        self.assertEqual(len(ctx.exception.prompt_hash), 16)
        self.assertGreaterEqual(ctx.exception.latency_ms, 0)


if __name__ == "__main__":
    unittest.main()
