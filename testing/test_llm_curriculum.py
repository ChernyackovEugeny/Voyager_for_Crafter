import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.curriculum import (
    Condition,
    CurriculumCall,
    HardcodedCurriculum,
    LLMCurriculum,
    Task,
)
from llm.curriculum_dsl import evaluate


def _info(achievements=None, inventory=None):
    return {
        "achievements": achievements if achievements is not None else {
            "collect_drink": 1,
            "eat_cow": 1,
            "place_table": 1,
        },
        "inventory": inventory or {"health": 9, "food": 9, "drink": 9},
        "player_pos": (3, 4),
        "semantic": None,
        "view_size": (9, 9),
    }


class CurriculumDslTests(unittest.TestCase):
    def test_all_operators(self):
        info = _info(inventory={"wood": 3})
        self.assertTrue(evaluate([Condition("inventory.wood", ">=", 3)], info))
        self.assertTrue(evaluate([Condition("inventory.wood", "<=", 3)], info))
        self.assertTrue(evaluate([Condition("inventory.wood", "==", 3)], info))
        self.assertTrue(evaluate([Condition("inventory.wood", ">", 2)], info))
        self.assertTrue(evaluate([Condition("inventory.wood", "<", 4)], info))

    def test_unknown_key_is_false(self):
        self.assertFalse(evaluate([Condition("inventory.gold", ">=", 1)], _info()))

    def test_multiple_conditions_are_and(self):
        info = _info(achievements={"collect_wood": 1}, inventory={"wood": 2})
        self.assertTrue(
            evaluate(
                [
                    Condition("inventory.wood", ">=", 2),
                    Condition("achievements.collect_wood", "==", 1),
                ],
                info,
            )
        )
        self.assertFalse(
            evaluate(
                [
                    Condition("inventory.wood", ">=", 3),
                    Condition("achievements.collect_wood", "==", 1),
                ],
                info,
            )
        )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(responses)
        )


class LLMCurriculumTests(unittest.TestCase):
    def _curriculum(self, responses, **kwargs):
        return LLMCurriculum(
            client=FakeClient(responses),
            max_retries=kwargs.pop("max_retries", 0),
            fallback=kwargs.pop("fallback", HardcodedCurriculum()),
            **kwargs,
        )

    def test_parses_valid_achievement_json(self):
        curriculum = self._curriculum([
            '{"name":"collect-wood","description":"Chop a tree.","achievement_key":"collect_wood","conditions":[]}'
        ])

        task = curriculum.propose_task(_info())

        self.assertEqual(task.achievement_key, "collect_wood")
        self.assertEqual(task.name, "collect-wood")

    def test_parses_sub_task_conditions(self):
        curriculum = self._curriculum([
            '{"name":"stockpile-wood","description":"Collect spare wood.","achievement_key":null,"conditions":[{"key":"inventory.wood","op":">=","value":4}]}'
        ])

        task = curriculum.propose_task(_info())

        self.assertIsNone(task.achievement_key)
        self.assertEqual(task.completion_conditions[0].key, "inventory.wood")
        self.assertTrue(
            curriculum.is_task_complete(task, _info(inventory={"wood": 4}))
        )

    def test_rejects_unknown_achievement_key_and_falls_back(self):
        curriculum = self._curriculum([
            '{"name":"bad","description":"Bad.","achievement_key":"collect_gold","conditions":[]}'
        ])

        task = curriculum.propose_task(_info())

        self.assertEqual(task.achievement_key, "collect_wood")

    def test_rejects_locked_achievement_and_falls_back(self):
        curriculum = self._curriculum([
            '{"name":"collect-stone","description":"Mine stone.","achievement_key":"collect_stone","conditions":[]}'
        ])

        task = curriculum.propose_task(_info())

        self.assertEqual(task.achievement_key, "collect_wood")

    def test_rejects_skip_list(self):
        curriculum = self._curriculum([
            '{"name":"collect-wood","description":"Chop a tree.","achievement_key":"collect_wood","conditions":[]}'
        ])

        task = curriculum.propose_task(_info(), skip={"collect_wood"})

        self.assertEqual(task.achievement_key, "collect_sapling")

    def test_rejects_empty_non_achievement_task(self):
        curriculum = self._curriculum([
            '{"name":"wander","description":"Wander.","achievement_key":null,"conditions":[]}'
        ])

        task = curriculum.propose_task(_info())

        self.assertEqual(task.achievement_key, "collect_wood")

    def test_rejects_already_complete_sub_task(self):
        curriculum = self._curriculum([
            '{"name":"stockpile-wood","description":"Collect spare wood.","achievement_key":null,"conditions":[{"key":"inventory.wood","op":">=","value":2}]}'
        ])

        task = curriculum.propose_task(_info(inventory={"wood": 2}))

        self.assertEqual(task.achievement_key, "collect_wood")

    def test_retry_with_correction_after_parse_error(self):
        client = FakeClient([
            'not json',
            '{"name":"collect-wood","description":"Chop a tree.","achievement_key":"collect_wood","conditions":[]}',
        ])
        curriculum = LLMCurriculum(client=client, max_retries=1)

        task = curriculum.propose_task(_info())

        self.assertEqual(task.achievement_key, "collect_wood")
        self.assertEqual(len(client.chat.completions.calls), 2)
        self.assertIn(
            "Previous output rejected",
            client.chat.completions.calls[1]["messages"][1]["content"],
        )

    def test_survival_check_happens_before_llm(self):
        client = FakeClient([
            '{"name":"collect-wood","description":"Chop a tree.","achievement_key":"collect_wood","conditions":[]}'
        ])
        curriculum = LLMCurriculum(client=client, max_retries=0)

        task = curriculum.propose_task(
            _info(inventory={"health": 4, "food": 9, "drink": 9})
        )

        self.assertEqual(task.name, "survive")
        self.assertEqual(client.chat.completions.calls, [])

    def test_record_task_failed_compacts_state(self):
        curriculum = self._curriculum([])
        task = Task("collect-wood", "Chop.", "collect_wood")
        curriculum.record_task_failed(
            task,
            {
                "obs": "large",
                "info": _info(
                    achievements={"collect_wood": 1},
                    inventory={"wood": 2, "stone": 0},
                ),
                "failure_reason": "stagnation",
                "executor_steps": 12,
                "executor_reason": "stagnation",
                "error_traceback": "ValueError: boom\ntrace",
            },
        )

        failure = curriculum.failures[0]
        self.assertEqual(failure.inventory_summary, {"wood": 2})
        self.assertEqual(failure.position, (3, 4))
        self.assertEqual(failure.error_first_line, "ValueError: boom")

    def test_only_last_n_failures_go_to_prompt(self):
        client = FakeClient([
            '{"name":"collect-wood","description":"Chop a tree.","achievement_key":"collect_wood","conditions":[]}'
        ])
        curriculum = LLMCurriculum(
            client=client,
            max_failures_in_context=1,
            max_retries=0,
        )
        curriculum.record_task_failed(Task("old-task", "Old.", None), {"info": _info()})
        curriculum.record_task_failed(Task("new-task", "New.", None), {"info": _info()})

        curriculum.propose_task(_info())

        prompt = client.chat.completions.calls[0]["messages"][1]["content"]
        self.assertNotIn("old-task", prompt)
        self.assertIn("new-task", prompt)

    def test_drain_pending_llm_calls_returns_and_clears(self):
        curriculum = self._curriculum([
            '{"name":"collect-wood","description":"Chop a tree.","achievement_key":"collect_wood","conditions":[]}'
        ])
        curriculum.propose_task(_info())

        calls = curriculum.drain_pending_llm_calls()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "curriculum")
        self.assertIsInstance(calls[0][1], CurriculumCall)
        self.assertEqual(curriculum.drain_pending_llm_calls(), ())


if __name__ == "__main__":
    unittest.main()
