import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.agent import Agent
from agent.executor import ExecutionResult, InterruptReason
from agent.memory import SpatialMemory
from agent.strategies import SkillSource
from llm.codegen import CodeGenCall
from llm.curriculum import Task
from skills.results import SkillCandidate
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
        self.calls = []
        self.pending_llm_calls = []
        self.completed = []

    def propose_task(self, info, *, skip=None):
        skip = skip or set()
        self.calls.append(set(skip))
        if info.get("achievements", {}).get("collect_wood"):
            return None
        for task in self.tasks:
            if task.skip_key not in skip:
                return task
        return None

    def is_task_complete(self, task, info):
        return bool(info.get("achievements", {}).get(task.achievement_key, 0))

    def record_task_failed(self, task, state_snapshot):
        self.failures.append((task, state_snapshot))

    def record_task_completed(self, task, state_snapshot=None):
        self.completed.append((task, state_snapshot))

    def drain_pending_llm_calls(self):
        calls = tuple(self.pending_llm_calls)
        self.pending_llm_calls = []
        return calls


class FakeSkillManager:
    def __init__(self, candidates=None):
        self.candidates = candidates or []
        self.retrieve_calls = []

    def retrieve(self, task_text):
        self.retrieve_calls.append(task_text)
        return self.candidates


class FakeRunLogger:
    def __init__(self):
        self.episodes = []
        self.llm_calls = []

    def record_episode(self, **kwargs):
        self.episodes.append(kwargs)
        return "episode-id"

    def record_llm_call(self, **kwargs):
        self.llm_calls.append(kwargs)


class FakeStrategy:
    def __init__(self, sources):
        self.sources = list(sources)
        self.unavailable = []
        self.completed = []
        self.failed = []
        self.candidate_names = []

    def acquire_skill(self, *, task, obs, info, candidates):
        self.candidate_names.append([candidate.skill.name for candidate in candidates])
        if self.sources:
            return self.sources.pop(0)
        return None

    def on_skill_unavailable(self, *, task, state, curriculum):
        self.unavailable.append(task.name)

    def on_task_completed(self, *, task, source, skill_manager):
        self.completed.append((task.name, source.reused_name))

    def on_task_failed(self, *, task, source, state, skill_manager, curriculum):
        self.failed.append((task.name, source.reused_name))
        curriculum.record_task_failed(task, state)

    def retrieval_route(self, candidates, task=None):
        return "test"

    @property
    def reuse_threshold(self):
        return 0.85


class FakeExecutor:
    def __init__(self, *, complete, reason=InterruptReason.COMPLETED, steps=1):
        self.complete = complete
        self.reason = reason
        self.steps = steps
        self.calls = []
        self.render_state_calls = 0

    def render_state(self, env):
        self.render_state_calls += 1

    def run(
        self,
        skill,
        env,
        initial_state,
        *,
        health_interrupt_enabled=True,
        danger_interrupt_enabled=False,
        survival_progress_enabled=False,
    ):
        self.calls.append({
            "skill": skill,
            "initial_state": initial_state,
            "health_interrupt_enabled": health_interrupt_enabled,
            "danger_interrupt_enabled": danger_interrupt_enabled,
            "survival_progress_enabled": survival_progress_enabled,
        })
        achievements = {"collect_wood": 1 if self.complete else 0}
        return ExecutionResult(
            reason=self.reason,
            steps=self.steps,
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


def _task(name="collect-wood", key="collect_wood"):
    return Task(
        name=name,
        description="Chop a tree to obtain wood.",
        achievement_key=key,
    )


def _skill_code():
    return "def collect_wood(state):\n    state = yield 0\n"


def _llm_call():
    return CodeGenCall(
        code=_skill_code(),
        raw_response=f"```python\n{_skill_code()}\n```",
        model="deepseek-v4-flash",
        prompt_template_id="codegen.v1",
        prompt_hash="abc123",
        prompt_tokens=100,
        prompt_cache_hit_tokens=40,
        prompt_cache_miss_tokens=60,
        completion_tokens=20,
        reasoning_tokens=None,
        latency_ms=123,
        cost_usd=0.001,
    )


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
    strategy=None,
    executor=None,
    max_iterations=5,
    run_logger=None,
):
    return Agent(
        env=FakeEnv(),
        curriculum=curriculum or FakeCurriculum([_task()]),
        skill_manager=skill_manager or FakeSkillManager(),
        strategy=strategy or FakeStrategy([SkillSource(code=_skill_code())]),
        executor=executor or FakeExecutor(complete=True),
        memory=SpatialMemory(),
        max_iterations_per_episode=max_iterations,
        run_logger=run_logger,
    )


class AgentTests(unittest.TestCase):
    def test_run_bootstraps_initial_info_with_noop_step(self):
        curriculum = FakeCurriculum([])
        agent = _agent(curriculum=curriculum)

        summary = agent.run()

        self.assertEqual(agent.env.actions, [0])
        self.assertEqual(summary["iterations"], 0)

    def test_executes_strategy_source_and_reports_completion(self):
        strategy = FakeStrategy([
            SkillSource(code=_skill_code(), reused_name="existing_wood")
        ])
        agent = _agent(strategy=strategy, executor=FakeExecutor(complete=True))

        summary = agent.run()

        self.assertEqual(summary["iterations"], 1)
        self.assertEqual(strategy.completed, [("collect-wood", "existing_wood")])
        self.assertEqual(strategy.failed, [])

    def test_failed_execution_calls_strategy_failure_hook(self):
        curriculum = FakeCurriculum([_task()])
        strategy = FakeStrategy([
            SkillSource(code=_skill_code(), reused_name="existing_wood")
        ])
        agent = _agent(
            curriculum=curriculum,
            strategy=strategy,
            executor=FakeExecutor(complete=False),
            max_iterations=1,
        )

        agent.run()

        self.assertEqual(strategy.failed, [("collect-wood", "existing_wood")])
        self.assertEqual(len(curriculum.failures), 1)

    def test_source_unavailable_skips_task_for_rest_of_episode(self):
        tasks = [
            _task("collect-wood", "collect_wood"),
            _task("collect-drink", "collect_drink"),
        ]
        curriculum = FakeCurriculum(tasks)
        strategy = FakeStrategy([
            None,
            SkillSource(code=_skill_code(), reused_name="drink"),
        ])
        agent = _agent(
            curriculum=curriculum,
            strategy=strategy,
            executor=FakeExecutor(complete=False),
            max_iterations=2,
        )

        summary = agent.run()

        self.assertEqual(strategy.unavailable, ["collect-wood"])
        self.assertIn("collect_wood", summary["skipped_tasks"])
        self.assertIn({"collect_wood"}, curriculum.calls)

    def test_load_failure_calls_strategy_failure_hook(self):
        curriculum = FakeCurriculum([_task()])
        strategy = FakeStrategy([
            SkillSource(code="def broken(state)\n    yield 0\n", reused_name="bad")
        ])
        agent = _agent(curriculum=curriculum, strategy=strategy)

        agent.run()

        self.assertEqual(strategy.failed, [("collect-wood", "bad")])

    def test_retrieves_candidates_before_strategy_acquire(self):
        manager = FakeSkillManager(candidates=[_candidate()])
        strategy = FakeStrategy([SkillSource(code=_skill_code())])
        agent = _agent(skill_manager=manager, strategy=strategy)

        agent.run()

        self.assertEqual(manager.retrieve_calls, ["Chop a tree to obtain wood."])

    def test_agent_logs_llm_call_from_generated_skill_source(self):
        run_logger = FakeRunLogger()
        strategy = FakeStrategy([
            SkillSource(code=_skill_code(), generated=True, llm_call=_llm_call())
        ])
        agent = _agent(
            strategy=strategy,
            executor=FakeExecutor(complete=True),
            run_logger=run_logger,
        )

        summary = agent.run(episode_num=7)

        self.assertEqual(summary["episode_id"], "episode-id")
        self.assertEqual(run_logger.llm_calls[0]["call_type"], "codegen")
        self.assertEqual(run_logger.llm_calls[0]["episode_num"], 7)
        self.assertEqual(run_logger.llm_calls[0]["model"], "deepseek-v4-flash")

    def test_agent_logs_curriculum_llm_call_after_proposal(self):
        run_logger = FakeRunLogger()
        curriculum = FakeCurriculum([_task()])
        curriculum.pending_llm_calls.append(("curriculum", _llm_call()))
        agent = _agent(
            curriculum=curriculum,
            executor=FakeExecutor(complete=True),
            run_logger=run_logger,
        )

        agent.run(episode_num=7)

        self.assertEqual(run_logger.llm_calls[0]["call_type"], "curriculum")
        self.assertEqual(run_logger.llm_calls[0]["episode_num"], 7)

    def test_agent_records_task_completed_on_curriculum(self):
        curriculum = FakeCurriculum([_task()])
        agent = _agent(curriculum=curriculum, executor=FakeExecutor(complete=True))

        agent.run()

        self.assertEqual(curriculum.completed[0][0].name, "collect-wood")

    def test_generated_zero_step_completion_is_rejected(self):
        strategy = FakeStrategy([
            SkillSource(code=_skill_code(), generated=True)
        ])
        agent = _agent(
            strategy=strategy,
            executor=FakeExecutor(complete=True, steps=0),
            max_iterations=1,
        )

        agent.run()

        self.assertEqual(strategy.completed, [])
        self.assertEqual(strategy.failed, [("collect-wood", None)])

    def test_max_iterations_stops_repeated_failures(self):
        curriculum = FakeCurriculum([_task()])
        strategy = FakeStrategy([
            SkillSource(code=_skill_code()),
            SkillSource(code=_skill_code()),
            SkillSource(code=_skill_code()),
        ])
        agent = _agent(
            curriculum=curriculum,
            strategy=strategy,
            executor=FakeExecutor(complete=False),
            max_iterations=3,
        )

        summary = agent.run()

        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(len(curriculum.failures), 3)

    def test_survive_failure_is_skipped_after_consecutive_limit(self):
        survive = _task("survive", None)
        collect = _task("collect-wood", "collect_wood")
        curriculum = FakeCurriculum([survive, collect])
        strategy = FakeStrategy([
            SkillSource(code=_skill_code()),
            SkillSource(code=_skill_code()),
            SkillSource(code=_skill_code()),
        ])
        agent = _agent(
            curriculum=curriculum,
            strategy=strategy,
            executor=FakeExecutor(complete=False),
            max_iterations=3,
        )
        agent.max_consecutive_survive_failures = 2

        summary = agent.run()

        self.assertEqual(summary["iterations"], 3)
        self.assertIn({"survive"}, curriculum.calls)
        self.assertEqual(strategy.failed[0][0], "survive")
        self.assertEqual(strategy.failed[1][0], "survive")
        self.assertEqual(strategy.failed[2][0], "collect-wood")

    def test_survive_disables_danger_interrupt_to_allow_recovery_policy(self):
        survive = _task("survive", None)
        executor = FakeExecutor(complete=False)
        agent = _agent(
            curriculum=FakeCurriculum([survive]),
            strategy=FakeStrategy([SkillSource(code=_skill_code())]),
            executor=executor,
            max_iterations=1,
        )

        agent.run()

        self.assertFalse(executor.calls[0]["danger_interrupt_enabled"])
        self.assertFalse(executor.calls[0]["health_interrupt_enabled"])
        self.assertTrue(executor.calls[0]["survival_progress_enabled"])

    def test_generic_task_skipped_after_consecutive_failures(self):
        collect = _task("collect-wood", "collect_wood")
        other = _task("collect-stone", "collect_stone")
        curriculum = FakeCurriculum([collect, collect, collect, other])
        strategy = FakeStrategy([
            SkillSource(code=_skill_code()),
            SkillSource(code=_skill_code()),
            SkillSource(code=_skill_code()),
            SkillSource(code=_skill_code()),
        ])
        agent = _agent(
            curriculum=curriculum,
            strategy=strategy,
            executor=FakeExecutor(complete=False),
            max_iterations=5,
        )
        agent.max_consecutive_task_failures = 3

        summary = agent.run()

        # After 3 failures, collect-wood should be in skipped, agent moves on.
        self.assertIn("collect_wood", summary["skipped_tasks"])
        self.assertEqual(strategy.failed[3][0], "collect-stone")

    def test_failed_reused_skill_is_blocked_for_same_task_in_episode(self):
        collect = _task("collect-wood", "collect_wood")
        manager = FakeSkillManager(candidates=[
            _candidate("bad_wood", 0.95),
            _candidate("backup_wood", 0.90),
        ])
        strategy = FakeStrategy([
            SkillSource(code=_skill_code(), reused_name="bad_wood"),
            SkillSource(code=_skill_code(), reused_name="backup_wood"),
        ])
        agent = _agent(
            curriculum=FakeCurriculum([collect, collect]),
            skill_manager=manager,
            strategy=strategy,
            executor=FakeExecutor(complete=False),
            max_iterations=2,
        )

        agent.run()

        self.assertEqual(strategy.candidate_names[0], ["bad_wood", "backup_wood"])
        self.assertEqual(strategy.candidate_names[1], ["backup_wood"])


if __name__ == "__main__":
    unittest.main()
