"""Business logic for skill persistence, retrieval, and metrics."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import EmbeddingConfig
from skills.description import compose_description
from skills.embedder import TextEmbedder
from skills.results import SaveResult, SkillCandidate
from storage.schemas import SkillRecord

logger = logging.getLogger(__name__)


class SkillManager:
    """Coordinates embeddings, deduplication, retrieval, and metric updates."""

    def __init__(
        self,
        *,
        repository: Any,
        embedder: TextEmbedder,
        config: EmbeddingConfig,
    ) -> None:
        self._repo = repository
        self._embedder = embedder
        self._cfg = config

    def save(
        self,
        *,
        name: str,
        code: str,
        task: str,
        deduplicate: bool = True,
    ) -> SaveResult:
        """Try to save a generated skill, rejecting duplicates."""
        if self._repo.get(name) is not None:
            logger.info("SkillManager.save: name %r already taken", name)
            return SaveResult(saved=False, outcome="name_taken")

        description = compose_description(name, task, code)
        embedding = self._embedder.encode(task)

        dup_name, dup_sim = self._most_similar_existing(embedding)
        if (
            deduplicate
            and dup_name is not None
            and dup_sim > self._cfg.similarity_dedup_threshold
        ):
            logger.info(
                "SkillManager.save: %r rejected as duplicate of %r (sim=%.3f)",
                name,
                dup_name,
                dup_sim,
            )
            return SaveResult(
                saved=False,
                outcome="duplicate",
                similar_to=dup_name,
                similarity=dup_sim,
            )

        skill = SkillRecord(name=name, code=code, description=description)
        self._repo.add(skill, self._normalize_vector(embedding))
        logger.info("SkillManager.save: %r added", name)
        return SaveResult(saved=True, outcome="ok", skill=skill)

    def retrieve(self, task_text: str, k: int | None = None) -> list[SkillCandidate]:
        """Return top-k similar skills for the task text."""
        limit = k if k is not None else self._cfg.top_k
        embedding = self._normalize_vector(self._embedder.encode(task_text))
        # Fetch a wider window, then rerank by observed reliability. Similarity
        # remains the public score so strategy thresholds keep their meaning.
        pairs = self._repo.search(embedding, k=max(limit * 3, limit))
        candidates = [SkillCandidate(skill=skill, similarity=sim) for skill, sim in pairs]
        candidates.sort(
            key=lambda candidate: (
                self._retrieval_score(candidate.skill, candidate.similarity),
                candidate.similarity,
            ),
            reverse=True,
        )
        return candidates[:limit]

    def exists(self, name: str) -> bool:
        """Return True if a skill with this repository name already exists."""
        return self._repo.get(name) is not None

    def record_success(self, name: str) -> None:
        self._safe_update(name, success_delta=1)

    def record_failure(self, name: str) -> None:
        self._safe_update(name, fail_delta=1)

    def update_episodic_score(self, name: str, score: float) -> None:
        self._safe_update(name, episodic_score=score)

    def get(self, name: str) -> SkillRecord | None:
        """Return a stored skill record by name."""
        return self._repo.get(name)

    def count(self) -> int:
        """Return number of stored skills when the repository supports it."""
        count = getattr(self._repo, "count", None)
        if count is None:
            skills = getattr(self._repo, "list_skills", None)
            if skills is None:
                return 0
            return len(skills())
        return int(count())

    def update_code(
        self,
        name: str,
        new_code: str,
        *,
        task: str | None = None,
        update_kind: str = "reflection",
    ) -> None:
        """Replace stored skill code while preserving description and embedding."""
        try:
            existing = self._repo.get(name)
            description = None
            embedding = None
            if existing is not None:
                base_task = task or existing.description.split(". Uses:", 1)[0]
                description = compose_description(name, base_task, new_code)
                embedding = self._normalize_vector(self._embedder.encode(base_task))
            try:
                self._repo.update_code(
                    name,
                    new_code,
                    description=description,
                    embedding=embedding,
                    reflected_delta=1 if update_kind == "reflection" else 0,
                    fix_delta=1 if update_kind == "fix" else 0,
                )
            except TypeError:
                self._repo.update_code(name, new_code)
                if embedding is not None:
                    self._repo.update_embedding(name, embedding)
        except KeyError:
            logger.warning(
                "SkillManager: skill %r missing on code update (skipping)",
                name,
            )

    @staticmethod
    def _retrieval_score(skill: SkillRecord, similarity: float) -> float:
        attempts = skill.success_count + skill.fail_count
        if attempts <= 0:
            return similarity
        reliability = (skill.success_count + 1.0) / (attempts + 2.0)
        fail_penalty = min(0.30, 0.04 * skill.fail_count)
        return similarity + 0.10 * (reliability - 0.5) - fail_penalty

    def _most_similar_existing(self, query: np.ndarray) -> tuple[str | None, float]:
        names, embeddings = self._repo.all_embeddings()
        if not names:
            return None, 0.0

        query_norm = self._normalize_vector(query)
        embedding_norms = self._normalize_matrix(embeddings)
        sims = embedding_norms @ query_norm
        idx = int(np.argmax(sims))
        return names[idx], float(sims[idx])

    def _safe_update(self, name: str, **deltas) -> None:
        try:
            self._repo.update_metrics(name, **deltas)
        except KeyError:
            logger.warning(
                "SkillManager: skill %r missing on metrics update (skipping)",
                name,
            )

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return vec / norm

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        mat = np.asarray(matrix, dtype=np.float32)
        if mat.size == 0:
            return mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / (norms + 1e-12)
