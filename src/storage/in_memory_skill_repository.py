"""In-memory skill library — same public API as ChromaSkillRepository.

Used in two places:
  1. Unit tests for SkillManager / Agent — no Chroma needed.
  2. Dev runs where Chroma is intentionally not started (`--no-chroma`).

Implements cosine similarity search directly on the stacked embedding matrix.
Performance is fine up to a few thousand skills; we are nowhere near that
scale, but the implementation is straight numpy so it's not a concern either.
"""
from __future__ import annotations

import logging

import numpy as np

from storage.schemas import SkillRecord

logger = logging.getLogger(__name__)


class InMemorySkillRepository:
    """Drop-in stand-in for ChromaSkillRepository."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillRecord] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        logger.info("InMemorySkillRepository initialised (no persistence)")

    # ------------------------------------------------------------------
    # Public API — must mirror ChromaSkillRepository
    # ------------------------------------------------------------------

    def add(self, skill: SkillRecord, embedding: np.ndarray) -> None:
        self._skills[skill.name] = skill
        self._embeddings[skill.name] = np.asarray(embedding, dtype=np.float32)

    def search(
        self, query_embedding: np.ndarray, k: int,
    ) -> list[tuple[SkillRecord, float]]:
        if not self._skills:
            return []

        names = list(self._skills.keys())
        embs = np.stack([self._embeddings[n] for n in names])  # (N, D)

        # Cosine similarity = (a · b) / (|a| |b|)
        embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        sims = embs_norm @ q_norm  # (N,)

        order = np.argsort(-sims)[:k]
        return [(self._skills[names[i]], float(sims[i])) for i in order]

    def get(self, name: str) -> SkillRecord | None:
        return self._skills.get(name)

    def list_skills(self) -> list[SkillRecord]:
        return sorted(self._skills.values(), key=lambda skill: skill.name)

    def delete(self, name: str) -> bool:
        if name not in self._skills:
            return False
        del self._skills[name]
        del self._embeddings[name]
        return True

    def update_metrics(
        self,
        name: str,
        *,
        success_delta: int = 0,
        fail_delta: int = 0,
        episodic_score: float | None = None,
    ) -> None:
        if name not in self._skills:
            raise KeyError(f"skill {name!r} not in library")
        existing = self._skills[name]
        updates = {
            "success_count": existing.success_count + success_delta,
            "fail_count": existing.fail_count + fail_delta,
        }
        if episodic_score is not None:
            updates["episodic_score"] = episodic_score
        self._skills[name] = existing.model_copy(update=updates)

    def update_code(self, name: str, new_code: str) -> None:
        if name not in self._skills:
            raise KeyError(f"skill {name!r} not in library")
        existing = self._skills[name]
        self._skills[name] = existing.model_copy(
            update={
                "code": new_code,
                "reflected_count": existing.reflected_count + 1,
            }
        )

    def update_embedding(self, name: str, embedding: np.ndarray) -> None:
        if name not in self._skills:
            raise KeyError(f"skill {name!r} not in library")
        self._embeddings[name] = np.asarray(embedding, dtype=np.float32)

    def all_embeddings(self) -> tuple[list[str], np.ndarray]:
        if not self._embeddings:
            return [], np.empty((0, 0), dtype=np.float32)
        names = list(self._embeddings.keys())
        stacked = np.stack([self._embeddings[n] for n in names])
        return names, stacked

    def count(self) -> int:
        return len(self._skills)
