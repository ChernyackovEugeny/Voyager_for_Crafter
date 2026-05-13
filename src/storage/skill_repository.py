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

    def list_skills(self) -> list[SkillRecord]:
        """Return all skills in this collection, sorted by name."""
        result = self._collection.get(include=["metadatas", "documents"])
        skills = [
            self._from_metadata(name, document, meta)
            for name, document, meta in zip(
                result["ids"],
                result["documents"],
                result["metadatas"],
            )
        ]
        return sorted(skills, key=lambda skill: skill.name)

    def delete(self, name: str) -> bool:
        """Delete one skill. Returns False if it was not present."""
        if self.get(name) is None:
            return False
        self._collection.delete(ids=[name])
        return True

    def export_entries(self) -> list[dict]:
        """Return JSON-serializable records including embeddings."""
        result = self._collection.get(
            include=["metadatas", "documents", "embeddings"],
        )
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = [None] * len(result["ids"])
        entries = []
        for name, document, meta, embedding in zip(
            result["ids"],
            result["documents"],
            result["metadatas"],
            embeddings,
        ):
            entries.append({
                "name": name,
                "description": document,
                "metadata": dict(meta),
                "embedding": (
                    np.asarray(embedding, dtype=np.float32).tolist()
                    if embedding is not None else None
                ),
            })
        return sorted(entries, key=lambda entry: entry["name"])

    def import_entries(self, entries: list[dict], *, overwrite: bool = False) -> int:
        """Import entries produced by export_entries(). Returns inserted count."""
        inserted = 0
        for entry in entries:
            name = entry["name"]
            if self.get(name) is not None:
                if not overwrite:
                    continue
                self._collection.delete(ids=[name])

            embedding = entry.get("embedding")
            if embedding is None:
                continue
            self._collection.add(
                ids=[name],
                embeddings=[np.asarray(embedding, dtype=np.float32).tolist()],
                documents=[entry["description"]],
                metadatas=[entry["metadata"]],
            )
            inserted += 1
        return inserted

    def count(self) -> int:
        return int(self._collection.count())

    def reset_collection(self) -> None:
        """Delete and recreate the current collection."""
        try:
            self._client.delete_collection(self._cfg.skills_collection)
        except Exception:
            logger.debug(
                "ChromaSkillRepository reset: collection did not exist: %s",
                self._cfg.skills_collection,
            )
        self._collection = self._client.get_or_create_collection(
            name=self._cfg.skills_collection,
            embedding_function=None,
            metadata=self._COLLECTION_METADATA,
        )

    def list_collections(self) -> list[str]:
        return sorted(
            collection.name if hasattr(collection, "name") else str(collection)
            for collection in self._client.list_collections()
        )

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

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
        updates = {
            "success_count": existing.success_count + success_delta,
            "fail_count": existing.fail_count + fail_delta,
        }
        if episodic_score is not None:
            updates["episodic_score"] = episodic_score
        updated = existing.model_copy(update=updates)
        self._collection.update(
            ids=[name],
            metadatas=[self._to_metadata(updated)],
        )

    def update_code(self, name: str, new_code: str) -> None:
        existing = self.get(name)
        if existing is None:
            raise KeyError(f"skill {name!r} not in library")
        updated = existing.model_copy(
            update={
                "code": new_code,
                "reflected_count": existing.reflected_count + 1,
            }
        )
        self._collection.update(
            ids=[name],
            metadatas=[self._to_metadata(updated)],
        )

    def update_embedding(self, name: str, embedding: np.ndarray) -> None:
        if self.get(name) is None:
            raise KeyError(f"skill {name!r} not in library")
        self._collection.update(
            ids=[name],
            embeddings=[embedding.astype(np.float32).tolist()],
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
            "reflected_count": int(s.reflected_count),
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
            reflected_count=int(meta.get("reflected_count", 0)),
            episodic_score=float(meta.get("episodic_score", 0.0)),
            created_at=datetime.fromisoformat(meta["created_at"]),
        )
