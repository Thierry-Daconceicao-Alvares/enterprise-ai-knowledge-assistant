"""Turn passages and questions into vectors.

- GeminiEmbedder: `gemini-embedding-001` through the Gemini API, with distinct
  task types for documents and questions, as recommended for retrieval.
- LocalHashEmbedder: a deterministic bag-of-words vector (hashed word stems)
  computed locally. It needs no API key and is used by the automated tests and
  for offline experiments. It only matches shared words, not meaning.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence
from functools import partial
from typing import Protocol

import numpy as np
from google import genai
from google.genai import types

from .errors import LLMResponseError
from .gemini import call
from .text import tokenize


class Embedder(Protocol):
    """Anything that turns texts into unit-length vectors."""

    @property
    def signature(self) -> str:
        """Identifies the model and its settings; vectors with different
        signatures cannot be compared, so each signature has its own index."""
        ...

    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, so that a dot product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


class GeminiEmbedder:
    """Embeddings from the Gemini API. Vectors of fewer than 3072 dimensions
    are not normalised by the API, so they are normalised here."""

    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        dimensions: int,
        batch_size: int = 50,
        max_retries: int = 3,
    ) -> None:
        self._client = client
        self.model_name = model_name
        self.dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    def signature(self) -> str:
        return f"{self.model_name}-{self.dimensions}"

    def _embed(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.dimensions
        )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            request = partial(
                self._client.models.embed_content,
                model=self.model_name,
                contents=batch,
                config=config,
            )
            response = call("embedding", request, self._max_retries)
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise LLMResponseError(
                    f"Expected {len(batch)} embeddings, received {len(embeddings)}."
                )
            vectors.extend(embedding.values or [] for embedding in embeddings)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dimensions:
            raise LLMResponseError(f"Unexpected embedding shape {matrix.shape}.")
        return l2_normalize(matrix)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        """Question vectors are cached: the same question asked twice (for
        example by the evaluation) costs one API call."""
        if text not in self._query_cache:
            if len(self._query_cache) >= 1024:
                self._query_cache.clear()
            self._query_cache[text] = self._embed([text], "RETRIEVAL_QUERY")[0]
        return self._query_cache[text]


class LocalHashEmbedder:
    """Hashed bag of word stems and word pairs (the "hashing trick").

    Two texts get similar vectors when they share words, which is enough to
    test the indexing and search mechanics without an API key. It is a lexical
    method: it does not know that "supplier" and "vendor" mean the same thing.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    @property
    def signature(self) -> str:
        return f"local-hash-{self.dimensions}"

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self.dimensions] += sign
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return l2_normalize(np.stack([self._vector(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))
