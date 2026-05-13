"""Data Transfer Objects passed across the storage boundary.

Pydantic models give us validation + typed serialization. The same SkillRecord
flies between SkillManager, ChromaSkillRepository, and InMemorySkillRepository
without any of them caring how the others store it.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SkillRecord(BaseModel):
    """Single source of truth for a skill in the library.

    `embedding` is intentionally NOT a field here — it travels alongside as a
    separate np.ndarray parameter on repo methods. Keeping it out of the
    Pydantic model avoids needless serialization/deserialization through JSON.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    code: str
    description: str
    success_count: int = 0
    fail_count: int = 0
    reflected_count: int = 0
    fix_count: int = 0
    episodic_score: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
