"""ChromaDB-backed skill library.

Design notes:
  * Embeddings are computed by SkillManager via sentence-transformers and
    passed in explicitly. Chroma's `embedding_function` is left as None so
    nothing happens server-side; we want one source of truth for the model.
  * Collection uses cosine similarity (hnsw:space=cosine). Chroma returns
    cosine *distance*, so similarity = 1 - distance.
  * Skill name is the primary key (Chroma id). Names come from the LLM and
    must be unique — SkillManager guarantees this before calling add().
  * All metadata stored in Chroma is plain JSON-serializable scalars; nested
    objects (timestamps, floats) are stringified/coerced.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaClientSettings

from config import ChromaConfig
from storage.schemas import SkillRecord

logger = logging.getLogger(__name__)


class ChromaSkillRepository:
    """Persistence layer over a remote ChromaDB instance.

    Public API:
        add(skill, embedding)
        search(query_embedding, k)   -> [(SkillRecord, similarity), ...]
        get(name)                    -> SkillRecord | None
        update_metrics(name, ...)    -> None
        all_embeddings()             -> (names, stacked_embeddings)
    """

    _COLLECTION_METADATA = {"hnsw:space": "cosine"}

    def __init__(self, cfg: ChromaConfig) -> None:
        self._cfg = cfg
        self._client = chromadb.HttpClient(
            host=cfg.host,
            port=cfg.port,
            settings=ChromaClientSettings(anonymized_telemetry=False),
        )
        # embedding_function=None — we pass pre-computed embeddings.
        self._collection = self._client.get_or_create_collection(
            name=cfg.skills_collection,
            embedding_function=None,
            metadata=self._COLLECTION_METADATA,
        )
        logger.info(
            "ChromaSkillRepository connected: host=%s port=%s collection=%s",
            cfg.host, cfg.port, cfg.skills_collection,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, skill: SkillRecord, embedding: np.ndarray) -> None:
        """Insert a skill. Caller is responsible for uniqueness of skill.name."""
        self._collection.add(
            ids=[skill.name],
            embeddings=[embedding.astype(np.float32).tolist()],
            documents=[skill.description],
            metadatas=[self._to_metadata(skill)],
        )

    def search(
        self, query_embedding: np.ndarray, k: int,
    ) -> list[tuple[SkillRecord, float]]:
        """Top-k similar skills, descending by cosine similarity in [-1, 1]."""
        result = self._collection.query(
            query_embeddings=[query_embedding.astype(np.float32).tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        ids = result["ids"][0]
        if not ids:
            return []

        out: list[tuple[SkillRecord, float]] = []
        for name, doc, meta, dist in zip(
            ids,
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            skill = self._from_metadata(name, doc, meta)
            similarity = 1.0 - float(dist)
            out.append((skill, similarity))
        return out

    def get(self, name: str) -> SkillRecord | None:
        result = self._collection.get(
            ids=[name],
            include=["metadatas", "documents"],
        )
        if not result["ids"]:
            return None
        return self._from_metadata(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
        )

    def update_metrics(
        self,
        name: str,
        *,
        success_delta: int = 0,
        fail_delta: int = 0,
        episodic_score: float | None = None,
    ) -> None:
        """Atomic-ish increment of skill metrics. KeyError if name unknown."""
        existing = self.get(name)
        if existing is None:
            raise KeyError(f"skill {name!r} not in library")
        existing.success_count += success_delta
        existing.fail_count += fail_delta
        if episodic_score is not None:
            existing.episodic_score = episodic_score
        self._collection.update(
            ids=[name],
            metadatas=[self._to_metadata(existing)],
        )

    def all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Used by SkillManager for batch dedup checks at save time."""
        result = self._collection.get(include=["embeddings"])
        ids: list[str] = result["ids"]
        embs = result["embeddings"]
        if not ids or embs is None or len(embs) == 0:
            return [], np.empty((0, 0), dtype=np.float32)
        return ids, np.asarray(embs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Metadata <-> SkillRecord conversion
    #
    # Chroma metadata values must be primitive scalars (str/int/float/bool)
    # — no nested dicts, no datetimes. We store created_at as ISO string.
    # ------------------------------------------------------------------

    @staticmethod
    def _to_metadata(s: SkillRecord) -> dict:
        return {
            "code": s.code,
            "success_count": int(s.success_count),
            "fail_count": int(s.fail_count),
            "episodic_score": float(s.episodic_score),
            "created_at": s.created_at.isoformat(),
        }

    @staticmethod
    def _from_metadata(name: str, document: str, meta: dict) -> SkillRecord:
        return SkillRecord(
            name=name,
            code=meta["code"],
            description=document,
            success_count=int(meta.get("success_count", 0)),
            fail_count=int(meta.get("fail_count", 0)),
            episodic_score=float(meta.get("episodic_score", 0.0)),
            created_at=datetime.fromisoformat(meta["created_at"]),
        )
