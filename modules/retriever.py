"""
retriever.py - question -> most relevant chunks.

Steps:  embed the question -> similarity search -> drop chunks whose score
is below MIN_SIMILARITY_SCORE.  If nothing survives, the app answers "I couldn't
find enough information in the uploaded document" instead of letting the LLM
guess (this is our first line of defence against hallucination).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from modules import embeddings
from modules.text_processor import Chunk
from modules.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    rank: int          # 1 = most similar
    score: float       # cosine similarity
    text: str
    metadata: dict     # source, page, chunk_id, language


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)  # only chunks above the threshold
    best_score: float = 0.0                                     # best score BEFORE filtering
    has_relevant_context: bool = False


def index_chunks(chunks: list[Chunk]) -> VectorStore:
    """Embed all chunks and build a fresh vector store (used on every new upload)."""
    vectors = embeddings.embed_passages([c.text for c in chunks])
    store = VectorStore()
    store.build(chunks, vectors)
    return store


class Retriever:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult:
        top_k = top_k or config.TOP_K
        min_score = config.MIN_SIMILARITY_SCORE if min_score is None else min_score

        if not question.strip() or not self.store.is_ready():
            return RetrievalResult()

        hits = self.store.search(embeddings.embed_query(question), top_k)
        if not hits:
            return RetrievalResult()

        relevant = [
            RetrievedChunk(rank=i, score=score, text=chunk.text, metadata=chunk.metadata)
            for i, (chunk, score) in enumerate(hits, start=1)
            if score >= min_score
        ]
        return RetrievalResult(
            chunks=relevant,
            best_score=hits[0][1],
            has_relevant_context=len(relevant) > 0,
        )


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Join retrieved chunks into one numbered text block for the LLM prompt."""
    blocks = []
    for c in chunks:
        page = c.metadata.get("page")
        where = f"{c.metadata.get('source', 'document')}" + (f", page {page}" if page else "")
        blocks.append(f"[Source {c.rank} | {where}]\n{c.text}")
    return "\n\n".join(blocks)
