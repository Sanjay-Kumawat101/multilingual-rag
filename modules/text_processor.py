"""
text_processor.py - clean the extracted text and split it into chunks.

Why chunk?   An LLM prompt has limited space, and one embedding for a whole book
             would blur every topic together. Small chunks = precise retrieval.
Why overlap? A sentence that falls on a chunk boundary would otherwise be cut in
             half and lose its context. Repeating the last ~150 characters of
             one chunk at the start of the next keeps that sentence intact.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from modules.document_loader import EmptyDocumentError, PageText

MIN_CHUNK_CHARS = 20  # ignore fragments such as page numbers

# The splitter tries these separators from left to right: paragraph -> line ->
# sentence end -> clause -> word -> character. "।" (danda) and "॥" (double danda)
# are the full stops of Hindi / Marathi / Sanskrit, so sentences stay whole.
SEPARATORS = ["\n\n", "\n", "। ", "॥ ", ". ", "? ", "! ", "; ", ", ", " ", ""]

# Characters to delete: zero-width space, BOM, soft hyphen, word joiner.
# NOTE: U+200C / U+200D (ZWNJ / ZWJ) are deliberately KEPT - they change how
# Devanagari conjuncts are shaped and are part of correct spelling.
_INVISIBLE = {ord(c): None for c in "\u200b\ufeff\u00ad\u2060"}
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class Chunk:
    text: str
    metadata: dict  # {"source", "page", "chunk_id", "language"}


def clean_text(text: str) -> str:
    """Normalize extracted text (PDF extraction is messy)."""
    # NFC makes the same Devanagari letter always use the same code points,
    # so the tokenizer / embedding model sees consistent input.
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)   # "infor-\nmation" -> "information"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)                   # keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)             # hard line-wraps -> space
    return text.strip()


def build_chunks(
    pages: list[PageText],
    source: str,
    language: str | None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Clean each page, split it into overlapping chunks and attach metadata."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if chunk_overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator="end",   # keep the "." or "।" at the end of its sentence
        length_function=len,
    )

    chunks: list[Chunk] = []
    chunk_id = 0
    # Chunking page by page keeps the page number in the metadata accurate.
    # (Trade-off: a sentence that runs across a page break is split.)
    for page in pages:
        cleaned = clean_text(page.text)
        for piece in splitter.split_text(cleaned):
            if len(piece.strip()) < MIN_CHUNK_CHARS:
                continue
            chunks.append(Chunk(
                text=piece.strip(),
                metadata={
                    "source": source,
                    "page": page.page,          # None when the format has no pages
                    "chunk_id": chunk_id,
                    "language": language or "Unknown",
                },
            ))
            chunk_id += 1

    if not chunks:
        raise EmptyDocumentError()
    return chunks
