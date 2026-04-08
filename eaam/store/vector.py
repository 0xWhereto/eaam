"""Vector store for semantic similarity search using ChromaDB."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eaam.config import EmbeddingConfig, VectorConfig

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store for semantic similarity search."""

    def __init__(self, vector_config: VectorConfig | None = None, embedding_config: EmbeddingConfig | None = None):
        from eaam.config import EmbeddingConfig, VectorConfig

        self.vector_config = vector_config or VectorConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self._collection = None
        self._client = None
        self._embedding_fn = None
        self._init_store()

    def _init_store(self):
        import chromadb

        persist_path = Path(self.vector_config.persist_path)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(persist_path))

        # Use sentence-transformers for embedding
        self._embedding_fn = _SentenceTransformerEmbedding(self.embedding_config.model)

        self._collection = self._client.get_or_create_collection(
            name="eaam_memories",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Vector store initialized: %d entries in %s",
            self._collection.count(),
            persist_path,
        )

    def add(self, memory_id: str, text: str, metadata: dict | None = None):
        """Add a text to the vector store."""
        self._collection.upsert(
            ids=[memory_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def query(self, text: str, n_results: int = 10) -> list[tuple[str, float]]:
        """Search for similar texts. Returns list of (memory_id, similarity_score)."""
        results = self._collection.query(
            query_texts=[text],
            n_results=n_results,
            include=["distances"],
        )

        ids = results["ids"][0] if results["ids"] else []
        # ChromaDB returns cosine distance; convert to similarity
        distances = results["distances"][0] if results["distances"] else []

        return [(mid, 1.0 - dist) for mid, dist in zip(ids, distances)]

    def query_by_embedding(self, embedding: list[float], n_results: int = 10) -> list[tuple[str, float]]:
        """Search by raw embedding vector."""
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["distances"],
        )

        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        return [(mid, 1.0 - dist) for mid, dist in zip(ids, distances)]

    def delete(self, memory_id: str):
        try:
            self._collection.delete(ids=[memory_id])
        except Exception:
            pass  # may not exist

    def count(self) -> int:
        return self._collection.count()


class _SentenceTransformerEmbedding:
    """ChromaDB-compatible embedding function using sentence-transformers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def name(self) -> str:
        return f"sentence-transformer-{self.model_name}"

    @classmethod
    def build_from_config(cls, config: dict) -> "_SentenceTransformerEmbedding":
        return cls(config["model_name"])

    def get_config(self) -> dict:
        return {"model_name": self.model_name}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)

    def __call__(self, input: list[str]) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)
