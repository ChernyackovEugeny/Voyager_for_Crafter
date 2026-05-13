import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.strategies.training import TrainingStrategy
from llm.codegen import CodeGenCall
from llm.curriculum import Task
from skills.runner import load_skill
from skills.results import SaveResult, SkillCandidate
from storage.schemas import SkillRecord


class FakeCodeGen:
    def __init__(self, code=None, fail=False):
        self.code = code or _skill_code()
        self.fail = fail
        self.calls = []

    def get_code(self, *, state_text, task, retrieved_skills, previous_failure=None):
        self.calls.append({
            "state_text": state_text,
            "task": task,
            "retrieved_skills": retrieved_skills,
            "previous_failure": previous_failure,
        })
        if self.fail:
            raise RuntimeError("llm down")
        return self._call(self.code, "codegen.v1")

    @staticmethod
    def _call(code, template_id):
        return _llm_call(code, template_id)


class FakeBugFixer:
    def __init__(self, fix_codes=None):
        self.fix_calls = []
        self.fix_codes = list(fix_codes or [])

    def fix(self, *, skill_code, error_traceback, state_text, task):
        self.fix_calls.append({
            "skill_code": skill_code,
            "error_traceback": error_traceback,
            "state_text": state_text,
            "task": task,
        })
        code = self.fix_codes.pop(0) if self.fix_codes else skill_code
        return _llm_call(code, "fix_bug.v1")


def _llm_call(code, template_id):
    return CodeGenCall(
        code=code,
        raw_response=f"```python\n{code}\n```",
        model="deepseek-v4-flash",
        prompt_template_id=template_id,
        prompt_hash="abc123",
        prompt_tokens=100,
        prompt_cache_hit_tokens=40,
        prompt_cache_miss_tokens=60,
        completion_tokens=20,
        reasoning_tokens=None,
        latency_ms=123,
        cost_usd=0.001,
    )


class FakeSkillManager:
    def __init__(self):
        self.saved = []
        self.successes = []
        self.failures = []
        self.existing = set()
        self.records = {}
        self.code_updates = []

    def save(self, *, name, code, task):
        self.saved.append({"name": name, "code": code, "task": task})
        self.existing.add(name)
        skill = SkillRecord(name=name, code=code, description=task)
        self.records[name] = skill
        return SaveResult(
            saved=True,
            outcome="ok",
            skill=skill,
        )

    def exists(self, name):
        return name in self.existing

    def record_success(self, name):
        self.successes.append(name)

    def record_failure(self, name):
        self.failures.append(name)

    def get(self, name):
        return self.records.get(name)

    def update_code(self, name, code):
        self.code_updates.append((name, code))
        self.records[name] = self.records[name].model_copy(
            update={
                "code": code,
                "reflected_count": self.records[name].reflected_count + 1,
            }
        )


class FakeReflectionCall:
    code = "def collect_wood(state):\n    state = yield 1\n"
    raw_response = f"```python\n{code}\n```"
    model = "deepseek-reasoner"
    prompt_template_id = "reflection.v1"
    prompt_hash = "def456"
    prompt_tokens = 100
    prompt_cache_hit_tokens = 20
    prompt_cache_miss_tokens = 80
    completion_tokens = 30
    reasoning_tokens = 10
    latency_ms = 500
    cost_usd = 0.01

    @property
    def tokens_in(self):
        return self.prompt_tokens

    @property
    def tokens_out(self):
        return self.completion_tokens


class FakeReflection:
    def __init__(self):
        self.calls = []
        self.call = FakeReflectionCall()

    def improve_skill(self, ctx):
        self.calls.append(ctx)
        return self.call


class DuplicateSkillManager(FakeSkillManager):
    def save(self, *, name, code, task):
        self.saved.append({"name": name, "code": code, "task": task})
        return SaveResult(
            saved=False,
            outcome="duplicate",
            similar_to="collect_wood",
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


def _survive_task():
    return Task(
        name="survive",
        description="Restore survival stats and avoid visible enemies.",
        achievement_key=None,
    )


def _place_table_task():
    return Task(
        name="place-table",
        description="Place a crafting table on the ground (consumes 2 wood).",
        achievement_key="place_table",
    )


def _skill_code():
    return "def collect_wood(state):\n    state = yield 0\n"


def _broken_skill_code():
    return "def collect_wood(state):\n    state = yield from move_left(state)\n"


def _candidate(name="collect_wood", similarity=0.9):
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

        self.assertEqual(source.reused_name, "collect_wood")
        self.assertFalse(source.generated)
        self.assertEqual(codegen.calls, [])

    def test_high_similarity_incompatible_achievement_does_not_reuse(self):
        codegen = FakeCodeGen()
        strategy = TrainingStrategy(codegen=codegen, reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_place_table_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(name="collect_wood_2", similarity=0.95)],
        )

        self.assertTrue(source.generated)
        self.assertEqual(len(codegen.calls), 1)

    def test_survive_does_not_reuse_narrow_survival_subskill_as_controller(self):
        codegen = FakeCodeGen()
        strategy = TrainingStrategy(codegen=codegen, reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_survive_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(name="collect_drink", similarity=0.95)],
        )

        self.assertTrue(source.generated)
        self.assertEqual(len(codegen.calls), 1)

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
            "collect_wood",
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

    def test_codegen_call_is_attached_to_generated_source(self):
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        self.assertIsNotNone(source)
        self.assertIsNotNone(source.llm_call)
        self.assertEqual(source.llm_call.model, "deepseek-v4-flash")
        self.assertEqual(source.llm_call.prompt_cache_hit_tokens, 40)

    def test_fix_bug_repairs_generated_skill_before_returning_source(self):
        codegen = FakeCodeGen(
            code=_broken_skill_code(),
        )
        bug_fixer = FakeBugFixer(fix_codes=[_skill_code()])
        strategy = TrainingStrategy(
            codegen=codegen,
            bug_fixer=bug_fixer,
            reuse_threshold=0.85,
            max_fix_attempts=3,
            skill_validator=lambda code: load_skill(code),
        )

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        self.assertIsNotNone(source)
        self.assertEqual(source.code, _skill_code())
        self.assertEqual(len(bug_fixer.fix_calls), 1)
        self.assertEqual(
            [call_type for call_type, _ in source.llm_calls],
            ["codegen", "fix_bug"],
        )
        self.assertEqual(source.llm_call_type, "fix_bug")

    def test_fix_bug_exhaustion_returns_none(self):
        codegen = FakeCodeGen(
            code="def collect_wood(state)\n    state = yield 0\n",
        )
        bug_fixer = FakeBugFixer(fix_codes=[
            _broken_skill_code(),
            "def collect_wood(state):\n    state = yield from move_right(state)\n",
        ])
        strategy = TrainingStrategy(
            codegen=codegen,
            bug_fixer=bug_fixer,
            reuse_threshold=0.85,
            max_fix_attempts=2,
            skill_validator=lambda code: load_skill(code),
        )

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        self.assertIsNone(source)
        self.assertEqual(len(bug_fixer.fix_calls), 2)
        self.assertEqual(
            [call_type for call_type, _ in strategy.drain_pending_llm_calls()],
            ["codegen", "fix_bug", "fix_bug"],
        )

    def test_fix_bug_identical_code_aborts_retry_loop(self):
        codegen = FakeCodeGen(
            code=_broken_skill_code(),
        )
        bug_fixer = FakeBugFixer(fix_codes=[_broken_skill_code()])
        strategy = TrainingStrategy(
            codegen=codegen,
            bug_fixer=bug_fixer,
            reuse_threshold=0.85,
            max_fix_attempts=3,
            skill_validator=lambda code: load_skill(code),
        )

        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        self.assertIsNone(source)
        self.assertEqual(len(bug_fixer.fix_calls), 1)

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

    def test_generated_achievement_skill_uses_achievement_key_for_name(self):
        manager = FakeSkillManager()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)
        task = Task(
            name="wood-task",
            description="Chop a tree to obtain wood.",
            achievement_key="collect_wood",
        )

        strategy.on_task_completed(
            task=task,
            source=strategy.acquire_skill(
                task=task,
                obs=None,
                info=_info(),
                candidates=[],
            ),
            skill_manager=manager,
        )

        self.assertEqual(manager.saved[0]["name"], "collect_wood")

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
        self.assertEqual(manager.successes, ["collect_wood"])

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

        self.assertEqual(manager.successes, ["collect_wood"])

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

        self.assertEqual(manager.failures, ["collect_wood"])
        self.assertEqual(len(curriculum.failures), 1)

    def test_reused_failure_reflects_and_updates_code_when_enabled(self):
        manager = FakeSkillManager()
        manager.records["collect_wood"] = SkillRecord(
            name="collect_wood",
            code=_skill_code(),
            description="Chop a tree to obtain wood.",
            success_count=1,
        )
        curriculum = FakeCurriculum()
        reflection = FakeReflection()
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            reuse_threshold=0.85,
            reflection=reflection,
            reflection_enabled=True,
            max_reflections_per_skill=3,
        )
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.9)],
        )

        call = strategy.on_task_failed(
            task=_task(),
            source=source,
            state={
                "info": _info(),
                "failure_reason": "timeout",
                "executor_steps": 50,
            },
            skill_manager=manager,
            curriculum=curriculum,
        )

        self.assertIs(call, reflection.call)
        self.assertEqual(len(reflection.calls), 1)
        self.assertEqual(manager.code_updates[0][0], "collect_wood")
        self.assertIn("yield 1", manager.code_updates[0][1])

    def test_place_table_can_reflect_when_missing_wood_runtime_resource(self):
        manager = FakeSkillManager()
        manager.records["place_table"] = SkillRecord(
            name="place_table",
            code=_skill_code(),
            description="Place a crafting table on the ground.",
            success_count=1,
        )
        reflection = FakeReflection()
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            reuse_threshold=0.85,
            reflection=reflection,
            reflection_enabled=True,
        )

        call = strategy.on_task_failed(
            task=_place_table_task(),
            source=type("Source", (), {
                "reused_name": "place_table",
                "code": _skill_code(),
            })(),
            state={
                "info": _info() | {"inventory": {"wood": 1}},
                "failure_reason": "stagnation",
                "executor_steps": 15,
            },
            skill_manager=manager,
            curriculum=FakeCurriculum(),
        )

        self.assertIs(call, reflection.call)

    def test_reused_failure_skips_reflection_after_limit(self):
        manager = FakeSkillManager()
        manager.records["collect_wood"] = SkillRecord(
            name="collect_wood",
            code=_skill_code(),
            description="Chop a tree to obtain wood.",
            success_count=1,
            reflected_count=3,
        )
        reflection = FakeReflection()
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            reuse_threshold=0.85,
            reflection=reflection,
            reflection_enabled=True,
            max_reflections_per_skill=3,
        )
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.9)],
        )

        call = strategy.on_task_failed(
            task=_task(),
            source=source,
            state={"info": _info(), "failure_reason": "timeout"},
            skill_manager=manager,
            curriculum=FakeCurriculum(),
        )

        self.assertIsNone(call)
        self.assertEqual(reflection.calls, [])
        self.assertEqual(manager.code_updates, [])

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

    def test_generated_survive_partial_progress_saves_skill(self):
        manager = FakeSkillManager()
        curriculum = FakeCurriculum()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)
        source = strategy.acquire_skill(
            task=_survive_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        strategy.on_task_failed(
            task=_survive_task(),
            source=source,
            state={
                "initial_info": {
                    "inventory": {
                        "health": 5,
                        "food": 4,
                        "drink": 3,
                        "energy": 6,
                    }
                },
                "info": _info()
                | {
                    "inventory": {
                        "health": 5,
                        "food": 4,
                        "drink": 8,
                        "energy": 5,
                    }
                },
                "failure_reason": "task_incomplete",
                "executor_reason": "completed",
                "executor_steps": 25,
            },
            skill_manager=manager,
            curriculum=curriculum,
        )

        self.assertEqual(manager.saved[0]["name"], "survive")
        self.assertEqual(manager.successes, ["survive"])
        self.assertEqual(len(curriculum.failures), 1)

    def test_generated_survive_death_does_not_save_partial_skill(self):
        manager = FakeSkillManager()
        strategy = TrainingStrategy(codegen=FakeCodeGen(), reuse_threshold=0.85)
        source = strategy.acquire_skill(
            task=_survive_task(),
            obs=None,
            info=_info(),
            candidates=[],
        )

        strategy.on_task_failed(
            task=_survive_task(),
            source=source,
            state={
                "initial_info": {"inventory": {"health": 5, "drink": 3}},
                "info": _info() | {"inventory": {"health": 0, "drink": 8}},
                "failure_reason": "episode_done",
                "executor_reason": "episode_done",
                "executor_steps": 100,
            },
            skill_manager=manager,
            curriculum=FakeCurriculum(),
        )

        self.assertEqual(manager.saved, [])

    def test_generated_runtime_error_queues_fixed_skill_for_retry(self):
        fixed_code = "def collect_wood(state):\n    state = yield 1\n"
        bug_fixer = FakeBugFixer(fix_codes=[fixed_code])
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            bug_fixer=bug_fixer,
            reuse_threshold=0.85,
            skill_validator=lambda code, **kwargs: load_skill(code, **kwargs),
        )
        task = _task()
        source = strategy.acquire_skill(
            task=task, obs=None, info=_info(), candidates=[],
        )

        call = strategy.on_task_failed(
            task=task,
            source=source,
            state={
                "info": _info(),
                "failure_reason": "error",
                "error_traceback": "NameError: scout_safely",
            },
            skill_manager=FakeSkillManager(),
            curriculum=FakeCurriculum(),
        )
        retry = strategy.acquire_skill(
            task=task, obs=None, info=_info(), candidates=[],
        )

        self.assertEqual(call[0], "fix_bug")
        self.assertEqual(retry.code, fixed_code)
        self.assertTrue(retry.generated)
        self.assertEqual(len(bug_fixer.fix_calls), 1)

    def test_reused_runtime_error_updates_skill_after_fix_validation(self):
        fixed_code = "def collect_wood(state):\n    state = yield 1\n"
        bug_fixer = FakeBugFixer(fix_codes=[fixed_code])
        manager = FakeSkillManager()
        manager.records["collect_wood"] = SkillRecord(
            name="collect_wood",
            code=_skill_code(),
            description="Chop a tree to obtain wood.",
            success_count=1,
        )
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            bug_fixer=bug_fixer,
            reuse_threshold=0.85,
            skill_validator=lambda code, **kwargs: load_skill(code, **kwargs),
        )
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.9)],
        )

        call = strategy.on_task_failed(
            task=_task(),
            source=source,
            state={
                "info": _info(),
                "failure_reason": "error",
                "error_traceback": "TypeError: bad runtime call",
            },
            skill_manager=manager,
            curriculum=FakeCurriculum(),
        )

        self.assertEqual(call[0], "fix_bug")
        self.assertEqual(manager.code_updates, [("collect_wood", fixed_code)])

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

    def test_reflection_skipped_when_executor_steps_below_threshold(self):
        manager = FakeSkillManager()
        manager.records["collect_wood"] = SkillRecord(
            name="collect_wood",
            code=_skill_code(),
            description="Chop a tree to obtain wood.",
            success_count=1,
        )
        reflection = FakeReflection()
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            reuse_threshold=0.85,
            reflection=reflection,
            reflection_enabled=True,
            max_reflections_per_skill=3,
            min_reflection_steps=3,
        )
        source = strategy.acquire_skill(
            task=_task(),
            obs=None,
            info=_info(),
            candidates=[_candidate(similarity=0.9)],
        )

        call = strategy.on_task_failed(
            task=_task(),
            source=source,
            state={
                "info": _info(),
                "failure_reason": "health_low",
                "executor_steps": 1,
            },
            skill_manager=manager,
            curriculum=FakeCurriculum(),
        )

        self.assertIsNone(call)
        self.assertEqual(reflection.calls, [])

    def test_zero_step_task_incomplete_skips_reflection(self):
        manager = FakeSkillManager()
        manager.records["survive"] = SkillRecord(
            name="survive",
            code=_skill_code(),
            description="Restore survival stats.",
            success_count=1,
        )
        reflection = FakeReflection()
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            reuse_threshold=0.85,
            reflection=reflection,
            reflection_enabled=True,
            min_reflection_steps=3,
        )

        call = strategy.on_task_failed(
            task=Task(
                name="survive",
                description="Restore survival stats.",
                achievement_key=None,
            ),
            source=type("Source", (), {
                "reused_name": "survive",
                "code": _skill_code(),
            })(),
            state={
                "info": _info(),
                "failure_reason": "task_incomplete",
                "executor_steps": 0,
            },
            skill_manager=manager,
            curriculum=FakeCurriculum(),
        )

        self.assertIsNone(call)
        self.assertEqual(reflection.calls, [])

    def test_survive_can_reflect_even_when_protected_and_no_prior_success(self):
        manager = FakeSkillManager()
        manager.records["survive"] = SkillRecord(
            name="survive",
            code=_skill_code(),
            description="Restore survival stats.",
            success_count=0,
        )
        reflection = FakeReflection()
        strategy = TrainingStrategy(
            codegen=FakeCodeGen(),
            reuse_threshold=0.85,
            reflection=reflection,
            reflection_enabled=True,
            protected_reflection_skill_names={"survive"},
        )

        call = strategy.on_task_failed(
            task=_survive_task(),
            source=type("Source", (), {
                "reused_name": "survive",
                "code": _skill_code(),
            })(),
            state={
                "info": _info() | {"inventory": {"health": 0}},
                "failure_reason": "episode_done",
                "executor_steps": 80,
            },
            skill_manager=manager,
            curriculum=FakeCurriculum(),
        )

        self.assertIs(call, reflection.call)
        self.assertEqual(len(reflection.calls), 1)
        self.assertEqual(manager.code_updates[0][0], "survive")

    def test_codegen_receives_previous_failure_on_retry(self):
        codegen = FakeCodeGen()
        strategy = TrainingStrategy(codegen=codegen, reuse_threshold=0.85)
        task = _task()

        source1 = strategy.acquire_skill(
            task=task, obs=None, info=_info(), candidates=[],
        )
        self.assertIsNone(codegen.calls[0]["previous_failure"])

        strategy.on_task_failed(
            task=task,
            source=source1,
            state={"info": _info(), "failure_reason": "health_low"},
            skill_manager=FakeSkillManager(),
            curriculum=FakeCurriculum(),
        )

        strategy.acquire_skill(
            task=task, obs=None, info=_info(), candidates=[],
        )
        prev = codegen.calls[1]["previous_failure"]
        self.assertIsNotNone(prev)
        self.assertEqual(prev[0], source1.code)
        self.assertEqual(prev[1], "health_low")

    def test_previous_failure_cleared_on_task_success(self):
        codegen = FakeCodeGen()
        strategy = TrainingStrategy(codegen=codegen, reuse_threshold=0.85)
        task = _task()

        source1 = strategy.acquire_skill(
            task=task, obs=None, info=_info(), candidates=[],
        )
        strategy.on_task_failed(
            task=task,
            source=source1,
            state={"info": _info(), "failure_reason": "timeout"},
            skill_manager=FakeSkillManager(),
            curriculum=FakeCurriculum(),
        )
        # Now mark success - buffer should clear.
        strategy.on_task_completed(
            task=task,
            source=source1,
            skill_manager=FakeSkillManager(),
        )

        strategy.acquire_skill(
            task=task, obs=None, info=_info(), candidates=[],
        )
        self.assertIsNone(codegen.calls[-1]["previous_failure"])


if __name__ == "__main__":
    unittest.main()
