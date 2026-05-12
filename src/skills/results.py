"""Public DTOs returned by SkillManager."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from storage.schemas import SkillRecord


SaveOutcome = Literal["ok", "duplicate", "name_taken"]


@dataclass(frozen=True)
class SaveResult:
    """Outcome of SkillManager.save()."""

    saved: bool
    outcome: SaveOutcome
    skill: SkillRecord | None = None
    similar_to: str | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class SkillCandidate:
    """One item returned by SkillManager.retrieve()."""

    skill: SkillRecord
    similarity: float
