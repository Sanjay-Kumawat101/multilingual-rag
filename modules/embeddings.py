"""
embeddings.py - turn text into vectors with a multilingual model.

An embedding is a list of numbers (here 384) that represents the MEANING of a
text. A multilingual model is trained so that sentences with the same meaning
land close together even when they are written in different languages:

    "What is AI?"  ~  "आर्टिफिशियल इंटेलिजेंस क्या है?"  (nearby vectors)

That is what makes cross-lingual retrieval possible without translating anything.

All vectors are L2-normalized (length 1), so the inner product of two vectors
equals their cosine similarity (1.0 = same meaning, ~0 = unrelated).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

import config


class EmbeddingError(Exception):
    """The embedding model could not be loaded."""


@lru_cache(maxsize=1)
def get_model():
    """Load the model once and reuse it (loading takes several seconds)."""
    try:
        # Imported here so the rest of the project can be imported (and tested)
        # even before the heavy PyTorch dependency is installed.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception as exc:
        raise EmbeddingError(
            f"⚠️ Could not load the embedding model '{config.EMBEDDING_MODEL}'. "
            "The first run needs an internet connection to download it "
            f"(a few hundred MB). Details: {exc}"
        ) from exc
    model.max_seq_length = min(config.EMBEDDING_MAX_SEQ_LENGTH, 512)
    return model


def _prefixes() -> tuple[str, str]:
    """E5-family models were trained with 'query: ' / 'passage: ' prefixes."""
    if "e5" in config.EMBEDDING_MODEL.lower():
        return "query: ", "passage: "
    return "", ""


def _encode(texts: list[str]) -> np.ndarray:
    vectors = get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,   # unit length -> inner product == cosine similarity
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed document chunks. Returns an array of shape (len(texts), dim)."""
    _, prefix = _prefixes()
    return _encode([prefix + t for t in texts])


def embed_query(question: str) -> np.ndarray:
    """Embed one question. Returns shape (1, dim), ready for vector search."""
    prefix, _ = _prefixes()
    return _encode([prefix + question])
