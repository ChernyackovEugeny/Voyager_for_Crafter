from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSource:
    """Code selected for execution and its library origin, if any."""

    code: str
    reused_name: str | None = None
    generated: bool = False
    llm_call: object | None = None
    llm_call_type: str = "codegen"
    llm_calls: tuple[tuple[str, object], ...] = ()
    extra_skills: tuple[tuple[str, str], ...] = ()
    allowed_skill_names: frozenset[str] = frozenset()


class AgentStrategy:
    """Base class for train/inference behavior around the shared Agent loop."""

    name = "base"

    def acquire_skill(
        self,
        *,
        task,
        obs,
        info: dict,
        candidates,
    ):
        """Return SkillSource for this task, or None when no skill is available."""
        raise NotImplementedError

    def on_skill_unavailable(
        self,
        *,
        task,
        state: dict,
        curriculum,
    ) -> None:
        """React when acquire_skill returns None."""
        raise NotImplementedError

    def on_task_completed(self, *, task, source: SkillSource, skill_manager) -> None:
        """React to verified task completion."""
        raise NotImplementedError

    def on_task_failed(
        self,
        *,
        task,
        source: SkillSource,
        state: dict,
        skill_manager,
        curriculum,
    ) -> None:
        """React to execution that did not complete the task."""
        raise NotImplementedError

    def retrieval_route(self, candidates, task=None) -> str:
        """Human-readable decision label for Agent logs."""
        raise NotImplementedError

    @property
    def reuse_threshold(self) -> float:
        """Similarity threshold used for reuse decisions."""
        raise NotImplementedError
