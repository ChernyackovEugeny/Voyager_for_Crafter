import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.agent import Agent
from agent.executor import ExecutionResult, InterruptReason
from agent.memory import SpatialMemory
from llm.curriculum import Task
from skills.results import SaveResult, SkillCandidate
from storage.schemas import SkillRecord


class FakeEnv:
    def __init__(self):
        self.actions = []

    def reset(self):
        return "reset_obs"

    def step(self, action):
        self.actions.append(action)
        return "step_obs", 0.0, False, False, _state_info()


class FakeCurriculum:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.failures = []

    def propose_task(self, info):
        if info.get("achievements", {}).get("collect_wood"):
            return None
        return self.tasks[0] if self.tasks else None

    def is_task_complete(self, task, info):
        return bool(info.get("achievements", {}).get(task.achievement_key, 0))

    def record_task_failed(self, task, state_snapshot):
        self.failures.append((task, state_snapshot))


class FakeSkillManager:
    def __init__(self, candidates=None):
        self.candidates = candidates or []
        self.saved = []
        self.successes = []
        self.failures = []
        self.existing = set()

    def retrieve(self, task_text):
        return self.candidates

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


class FakeCodeGen:
    def __init__(self, code=None):
        self.code = code or _skill_code()
        self.calls = []

    def get_code(self, *, state_text, task, retrieved_skills):
        self.calls.append({
            "state_text": state_text,
            "task": task,
            "retrieved_skills": retrieved_skills,
        })
        return self.code


class FakeExecutor:
    def __init__(self, *, complete, reason=InterruptReason.COMPLETED):
        self.complete = complete
        self.reason = reason
        self.calls = []
        self.render_state_calls = 0

    def render_state(self, env):
        self.render_state_calls += 1

    def run(self, skill, env, initial_state):
        self.calls.append((skill, initial_state))
        achievements = {"collect_wood": 1 if self.complete else 0}
        return ExecutionResult(
            reason=self.reason,
            steps=1,
            total_reward=0.0,
            final_state={"obs": "final_obs", "info": _state_info(achievements)},
        )


def _state_info(achievements=None):
    return {
        "inventory": {"health": 9},
        "achievements": achievements or {"collect_wood": 0},
        "semantic": None,
        "player_pos": (0, 0),
        "view_size": (9, 7),
    }


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


def _agent(
    *,
    curriculum=None,
    skill_manager=None,
    codegen=None,
    executor=None,
    reuse_threshold=0.85,
    max_iterations=5,
):
    return Agent(
        env=FakeEnv(),
        curriculum=curriculum or FakeCurriculum([_task()]),
        skill_manager=skill_manager or FakeSkillManager(),
        codegen=codegen or FakeCodeGen(),
        executor=executor or FakeExecutor(complete=True),
        memory=SpatialMemory(),
        reuse_threshold=reuse_threshold,
        max_iterations_per_episode=max_iterations,
    )


class AgentTests(unittest.TestCase):
    def test_run_bootstraps_initial_info_with_noop_step(self):
        curriculum = FakeCurriculum([])
        agent = _agent(curriculum=curriculum)

        summary = agent.run()

        self.assertEqual(agent.env.actions, [0])
        self.assertEqual(summary["iterations"], 0)

    def test_generates_and_saves_new_skill_after_task_success(self):
        manager = FakeSkillManager()
        codegen = FakeCodeGen()
        agent = _agent(
            skill_manager=manager,
            codegen=codegen,
            executor=FakeExecutor(complete=True),
        )

        summary = agent.run()

        self.assertEqual(summary["iterations"], 1)
        self.assertEqual(len(codegen.calls), 1)
        self.assertEqual(manager.saved[0]["name"], "collect_wood")
        self.assertEqual(manager.successes, ["collect_wood"])
        self.assertEqual(manager.failures, [])

    def test_reuses_high_similarity_skill_and_records_success(self):
        manager = FakeSkillManager(candidates=[_candidate(similarity=0.9)])
        codegen = FakeCodeGen()
        agent = _agent(
            skill_manager=manager,
            codegen=codegen,
            executor=FakeExecutor(complete=True),
        )

        agent.run()

        self.assertEqual(codegen.calls, [])
        self.assertEqual(manager.saved, [])
        self.assertEqual(manager.successes, ["existing_wood"])

    def test_low_similarity_candidate_is_passed_to_codegen_context(self):
        manager = FakeSkillManager(candidates=[_candidate(similarity=0.2)])
        codegen = FakeCodeGen()
        agent = _agent(
            skill_manager=manager,
            codegen=codegen,
            executor=FakeExecutor(complete=True),
        )

        agent.run()

        self.assertEqual(len(codegen.calls), 1)
        self.assertEqual(
            codegen.calls[0]["retrieved_skills"][0]["name"],
            "existing_wood",
        )

    def test_reused_skill_failure_records_metric_and_task_failure(self):
        curriculum = FakeCurriculum([_task()])
        manager = FakeSkillManager(candidates=[_candidate(similarity=0.9)])
        agent = _agent(
            curriculum=curriculum,
            skill_manager=manager,
            executor=FakeExecutor(complete=False),
            max_iterations=1,
        )

        agent.run()

        self.assertEqual(manager.failures, ["existing_wood"])
        self.assertEqual(len(curriculum.failures), 1)

    def test_new_skill_failure_is_not_saved(self):
        curriculum = FakeCurriculum([_task()])
        manager = FakeSkillManager()
        agent = _agent(
            curriculum=curriculum,
            skill_manager=manager,
            executor=FakeExecutor(complete=False),
            max_iterations=1,
        )

        agent.run()

        self.assertEqual(manager.saved, [])
        self.assertEqual(len(curriculum.failures), 1)

    def test_duplicate_save_records_success_on_existing_skill(self):
        manager = DuplicateSkillManager()
        agent = _agent(
            skill_manager=manager,
            executor=FakeExecutor(complete=True),
        )

        agent.run()

        self.assertEqual(manager.successes, ["existing_wood"])

    def test_unique_skill_name_adds_version_suffix(self):
        manager = FakeSkillManager()
        manager.existing.update({"collect_wood", "collect_wood_v2"})
        agent = _agent(
            skill_manager=manager,
            executor=FakeExecutor(complete=True),
        )

        agent.run()

        self.assertEqual(manager.saved[0]["name"], "collect_wood_v3")

    def test_max_iterations_stops_repeated_failures(self):
        curriculum = FakeCurriculum([_task()])
        manager = FakeSkillManager()
        agent = _agent(
            curriculum=curriculum,
            skill_manager=manager,
            executor=FakeExecutor(complete=False),
            max_iterations=3,
        )

        summary = agent.run()

        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(len(curriculum.failures), 3)


if __name__ == "__main__":
    unittest.main()
