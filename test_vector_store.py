"""
test_vector_store.py - offline test (no model download needed).

It replaces the real embedding model with a tiny "bag of words" fake, so it only
verifies the ChromaDB + retriever logic (build, search, threshold, save/load, clear).
The real cross-lingual test is test_retrieval.py.

Run:  python test_vector_store.py
"""
import hashlib
import tempfile
from pathlib import Path

import numpy as np

from modules import embeddings
from modules.retriever import Retriever, format_context, index_chunks
from modules.text_processor import Chunk
from modules.vector_store import VectorStore

DIM = 256


def _fake_encode(texts):
    out = np.zeros((len(texts), DIM), dtype="float32")
    for row, text in enumerate(texts):
        for word in text.lower().replace("?", " ").replace(".", " ").split():
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


embeddings._encode = _fake_encode  # monkeypatch: skip the real model

TEXTS = [
    "Machine learning lets computers learn patterns from data.",
    "The Taj Mahal is a marble mausoleum in Agra built by Shah Jahan.",
    "Vector databases store embeddings and find similar items quickly.",
]
chunks = [Chunk(t, {"source": "demo.txt", "page": i + 1 if i else None, "chunk_id": i, "language": "English"})
          for i, t in enumerate(TEXTS)]

store = index_chunks(chunks)
retriever = Retriever(store)

res = retriever.retrieve("Who built the Taj Mahal in Agra?", top_k=2, min_score=0.1)
assert res.has_relevant_context and "Taj Mahal" in res.chunks[0].text, res
print("PASS  top-1 chunk is the Taj Mahal one, score =", round(res.chunks[0].score, 2))

res = retriever.retrieve("quantum spaceship banana", top_k=2, min_score=0.3)
assert not res.has_relevant_context and res.chunks == []
print("PASS  unrelated question -> no relevant context (best score", round(res.best_score, 2), ")")

assert len(retriever.retrieve("machine learning data", top_k=99, min_score=0.0).chunks) == 3
print("PASS  top_k larger than the index is clamped safely")

with tempfile.TemporaryDirectory() as tmp:
    store.save(Path(tmp))
    loaded = VectorStore.load(Path(tmp))
    again = Retriever(loaded).retrieve("Who built the Taj Mahal?", top_k=1, min_score=0.1)
    assert again.chunks[0].text == TEXTS[1]
print("PASS  save -> load round trip")

store.clear()
assert not store.is_ready() and Retriever(store).retrieve("anything").chunks == []
print("PASS  clear() empties the index")

print("\nformat_context preview:\n" + format_context(res.chunks or Retriever(index_chunks(chunks)).retrieve("Taj Mahal", 2, 0.1).chunks))
