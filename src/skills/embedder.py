"""Sentence-transformers wrapper for skill retrieval."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Sentence-level embedder returning float32 vectors."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        logger.info("Embedding model ready (dim=%d)", self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vec, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        mat = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(mat, dtype=np.float32)
