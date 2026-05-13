import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.strategies.inference import InferenceStrategy
from llm.curriculum import Task
from skills.results import SkillCandidate
from storage.schemas import SkillRecord


class FakeSkillManager:
    def __init__(self):
        self.successes = []
        self.failures = []
        self.saved = []

    def record_success(self, name):
        self.successes.append(name)

    def record_failure(self, name):
        self.failures.append(name)

    def save(self, *, name, code, task):
        self.saved.append((name, code, task))


class FakeCurriculum:
    def __init__(self):
        self.failures = []

    def record_task_failed(self, task, state_snapshot):
        self.failures.append((task, state_snapshot))


def _task():
    return Task(
        name="collect-wood",
        description="Chop a tree to obtain wood.",
        achievement_key="collect_wood",
    )


def _candidate(similarity, name="collect_wood"):
    skill = SkillRecord(
        name=name,
        code="def collect_wood(state):\n    state = yield 0\n",
        description="Chop a tree to obtain wood.",
    )
    return SkillCandidate(skill=skill, similarity=similarity)


class InferenceStrategyTests(unittest.TestCase):
    def test_high_similarity_reuses_skill(self):
        strategy = InferenceStrategy(reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info={},
            candidates=[_candidate(0.9)],
        )

        self.assertEqual(source.reused_name, "collect_wood")
        self.assertFalse(source.generated)

    def test_low_similarity_returns_none(self):
        strategy = InferenceStrategy(reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info={},
            candidates=[_candidate(0.2)],
        )

        self.assertIsNone(source)

    def test_high_similarity_incompatible_achievement_returns_none(self):
        strategy = InferenceStrategy(reuse_threshold=0.85)
        task = Task(
            name="place-table",
            description="Place a crafting table.",
            achievement_key="place_table",
        )

        source = strategy.acquire_skill(
            task=task,
            obs=None,
            info={},
            candidates=[_candidate(0.95, name="collect_wood_2")],
        )

        self.assertIsNone(source)

    def test_no_candidates_returns_none(self):
        strategy = InferenceStrategy(reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info={},
            candidates=[],
        )

        self.assertIsNone(source)

    def test_hooks_do_not_mutate_skill_manager_or_curriculum(self):
        strategy = InferenceStrategy(reuse_threshold=0.85)
        manager = FakeSkillManager()
        curriculum = FakeCurriculum()
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info={},
            candidates=[_candidate(0.9)],
        )

        strategy.on_task_completed(
            task=_task(),
            source=source,
            skill_manager=manager,
        )
        strategy.on_task_failed(
            task=_task(),
            source=source,
            state={},
            skill_manager=manager,
            curriculum=curriculum,
        )
        strategy.on_skill_unavailable(
            task=_task(),
            state={},
            curriculum=curriculum,
        )

        self.assertEqual(manager.successes, [])
        self.assertEqual(manager.failures, [])
        self.assertEqual(manager.saved, [])
        self.assertEqual(curriculum.failures, [])


if __name__ == "__main__":
    unittest.main()
