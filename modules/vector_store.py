"""
vector_store.py - ChromaDB-backed vector database.

ChromaDB stores the embeddings, chunk text, and metadata.
"""

from __future__ import annotations

from pathlib import Path
import uuid

import chromadb
import numpy as np

import config
from modules.text_processor import Chunk


COLLECTION_NAME = "multilingual_rag"


class VectorStore:
    def __init__(self) -> None:
        self.client = None
        self.collection = None
        self.chunks: list[Chunk] = []

        self._initialize()

    # --------------------------------------------------------------- setup

    def _initialize(self) -> None:
        """Initialize the persistent ChromaDB client."""

        folder = Path(config.VECTORSTORE_DIR)
        folder.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(folder)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    # ---------------------------------------------------------------- state

    @property
    def size(self) -> int:
        """Return number of stored chunks."""
        return self.collection.count()

    def is_ready(self) -> bool:
        """Return True when ChromaDB contains chunks."""
        return self.collection.count() > 0

    def clear(self) -> None:
        """Delete all stored chunks."""

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            }
        )

        self.chunks = []

    # ---------------------------------------------------------------- build

    def build(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """
        Replace the current vector database with a new one.
        """

        if len(chunks) == 0:
            raise ValueError(
                "Cannot build an index from zero chunks."
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        # Clear previous collection.
        self.clear()

        vectors = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Convert embeddings to Python lists because ChromaDB
        # expects standard Python lists.
        embedding_list = vectors.tolist()

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            chunk.metadata
            for chunk in chunks
        ]

        # Chroma requires metadata values to be simple types.
        cleaned_metadatas = []

        for metadata in metadatas:
            cleaned = {}

            for key, value in metadata.items():

                if value is None:
                    cleaned[key] = ""

                elif isinstance(
                    value,
                    (str, int, float, bool)
                ):
                    cleaned[key] = value

                else:
                    cleaned[key] = str(value)

            cleaned_metadatas.append(cleaned)

        ids = [
            f"chunk-{uuid.uuid4().hex}"
            for _ in chunks
        ]

        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=cleaned_metadatas
        )

        self.chunks = list(chunks)

    # ---------------------------------------------------------------- search

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int
    ) -> list[tuple[Chunk, float]]:
        """
        Search ChromaDB for the most similar chunks.

        Returns:
            list of (Chunk, similarity_score)
        """

        if not self.is_ready():
            return []

        k = max(
            1,
            min(top_k, self.collection.count())
        )

        query = np.asarray(
            query_vector,
            dtype=np.float32
        )

        if query.ndim == 1:
            query = query.tolist()
        else:
            query = query[0].tolist()

        results = self.collection.query(
            query_embeddings=[query],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):
            chunk = Chunk(
                text=document,
                metadata=metadata or {}
            )

            # ChromaDB cosine distance:
            #
            # distance = 1 - cosine_similarity
            #
            # Convert it back to similarity score.
            similarity = 1.0 - float(distance)

            output.append(
                (chunk, similarity)
            )

        return output

    # ------------------------------------------------------------ persistence

    def save(
        self,
        folder: Path | None = None
    ) -> None:
        """
        ChromaDB is persistent by default.

        Therefore no separate index file needs to be written.
        """

        if not self.is_ready():
            raise ValueError(
                "Nothing to save: the vector store is empty."
            )

        # PersistentClient automatically saves the database.
        return None

    @classmethod
    def load(
        cls,
        folder: Path | None = None
    ) -> "VectorStore":
        """
        Load an existing persistent ChromaDB database.
        """

        store = cls()

        if not store.is_ready():
            raise FileNotFoundError(
                "No saved ChromaDB collection found."
            )

        return store
