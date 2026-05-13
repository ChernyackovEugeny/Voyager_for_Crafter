import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.strategies.training import TrainingStrategy
from llm.curriculum import Task
from skills.results import SaveResult, SkillCandidate
from storage.schemas import SkillRecord


class FakeCodeGen:
    def __init__(self, code=None, fail=False):
        self.code = code or _skill_code()
        self.fail = fail
        self.calls = []

    def get_code(self, *, state_text, task, retrieved_skills):
        self.calls.append({
            "state_text": state_text,
            "task": task,
            "retrieved_skills": retrieved_skills,
        })
        if self.fail:
            raise RuntimeError("llm down")
        return self.code


class FakeSkillManager:
    def __init__(self):
        self.saved = []
        self.successes = []
        self.failures = []
        self.existing = set()

    def save(self, *, name, code, task):
        self.saved.append({"name": name, "code": code, "task": task})
        self.existing.add(name)
        return SaveResult(
            saved=True,
            outcome="ok",
            skill=SkillRecord(name=name, code=code, description=task),
        )

    def exists(self, name):
        return name in self.existing

    def record_success(self, name):
        self.successes.append(name)

    def record_failure(self, name):
        self.failures.append(name)


class DuplicateSkillManager(FakeSkillManager):
    def save(self, *, name, code, task):
        self.saved.append({"name": name, "code": code, "task": task})
        return SaveResult(
            saved=False,
            outcome="duplicate",
            similar_to="existing_wood",
            similarity=0.95,
        )


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


def _skill_code():
    return "def collect_wood(state):\n    state = yield 0\n"


def _candidate(name="existing_wood", similarity=0.9):
    skill = SkillRecord(
        name=name,
        code=_skill_code(),
        description="Chop a tree to obtain wood.",
    )
    return SkillCandidate(skill=skill, similarity=similarity)


def _info():
    return {
        "inventory": {},
        "achievements": {},
        "semantic": None,
        "player_pos": (0, 0),
        "view_size": (9, 7),
    }


class TrainingStrategyTests(unittest.TestCase):
    def test_high_similarity_reuses_without_codegen(self):
        codegen = FakeCodeGen()
        strategy = TrainingStrategy(codegen=codegen, reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.9)],
        )

        self.assertEqual(source.reused_name, "existing_wood")
        self.assertFalse(source.generated)
        self.assertEqual(codegen.calls, [])

    def test_low_similarity_calls_codegen_with_retrieved_context(self):
        codegen = FakeCodeGen()
        strategy = TrainingStrategy(codegen=codegen, reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.2)],
        )

        self.assertTrue(source.generated)
        self.assertEqual(len(codegen.calls), 1)
        self.assertEqual(
            codegen.calls[0]["retrieved_skills"][0]["name"],
            "existing_wood",
        )

    def test_codegen_error_returns_none(self):
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(fail=True),
            reuse_threshold=0.85,
        )

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        self.assertIsNone(source)

    def test_generated_success_saves_and_records_success(self):
        manager = FakeSkillManager()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)

        strategy.on_task_completed(
            task=_task(),
            source=strategy.acquire_skill(
                task=_task(),
                obs=None,
                info=_info(),
                candidates=[],
            ),
            skill_manager=manager,
        )

        self.assertEqual(manager.saved[0]["name"], "collect_wood")
        self.assertEqual(manager.successes, ["collect_wood"])

    def test_reused_success_records_success_only(self):
        manager = FakeSkillManager()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.9)],
        )

        strategy.on_task_completed(
            task=_task(),
            source=source,
            skill_manager=manager,
        )

        self.assertEqual(manager.saved, [])
        self.assertEqual(manager.successes, ["existing_wood"])

    def test_duplicate_save_records_success_on_existing_skill(self):
        manager = DuplicateSkillManager()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)

        strategy.on_task_completed(
            task=_task(),
            source=strategy.acquire_skill(
                task=_task(),
                obs=None,
                info=_info(),
                candidates=[],
            ),
            skill_manager=manager,
        )

        self.assertEqual(manager.successes, ["existing_wood"])

    def test_reused_failure_records_metric_and_curriculum_failure(self):
        manager = FakeSkillManager()
        curriculum = FakeCurriculum()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.9)],
        )

        strategy.on_task_failed(
            task=_task(),
            source=source,
            state={"info": _info()},
            skill_manager=manager,
            curriculum=curriculum,
        )

        self.assertEqual(manager.failures, ["existing_wood"])
        self.assertEqual(len(curriculum.failures), 1)

    def test_generated_failure_does_not_save(self):
        manager = FakeSkillManager()
        curriculum = FakeCurriculum()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        strategy.on_task_failed(
            task=_task(),
            source=source,
            state={"info": _info()},
            skill_manager=manager,
            curriculum=curriculum,
        )

        self.assertEqual(manager.saved, [])
        self.assertEqual(len(curriculum.failures), 1)

    def test_unique_skill_name_adds_version_suffix(self):
        manager = FakeSkillManager()
        manager.existing.update({"collect_wood", "collect_wood_v2"})
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)

        strategy.on_task_completed(
            task=_task(),
            source=strategy.acquire_skill(
                task=_task(),
                obs=None,
                info=_info(),
                candidates=[],
            ),
            skill_manager=manager,
        )

        self.assertEqual(manager.saved[0]["name"], "collect_wood_v3")


if __name__ == "__main__":
    unittest.main()
