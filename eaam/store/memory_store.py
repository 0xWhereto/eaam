"""Unified memory store combining graph and vector stores."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from eaam.models import Edge, EdgeType, Memory, VAD
from eaam.store.graph import GraphStore
from eaam.store.vector import VectorStore

if TYPE_CHECKING:
    from eaam.config import EAAMConfig

logger = logging.getLogger(__name__)


class MemoryStore:
    """Unified interface over graph + vector stores."""

    def __init__(self, config: EAAMConfig | None = None):
        from eaam.config import EAAMConfig

        self.config = config or EAAMConfig()
        self.graph = GraphStore(self.config.graph)
        self.vector = VectorStore(self.config.vector, self.config.embedding)

    def add(self, memory: Memory):
        """Store a memory in both graph and vector stores."""
        self.graph.add_memory(memory)
        self.vector.add(
            memory_id=memory.id,
            text=memory.content,
            metadata={
                "conversation_id": memory.conversation_id,
                "valence": memory.emotion.valence,
                "arousal": memory.emotion.arousal,
                "dominance": memory.emotion.dominance,
                "role": memory.role,
            },
        )

    def get(self, memory_id: str) -> Memory | None:
        return self.graph.get_memory(memory_id)

    def add_edge(self, edge: Edge):
        if not self.graph.edge_exists(edge.source_id, edge.target_id, edge.edge_type):
            self.graph.add_edge(edge)

    def semantic_search(self, query: str, n: int = 20) -> list[tuple[str, float]]:
        """Find semantically similar memories. Returns (memory_id, score) pairs."""
        return self.vector.query(query, n_results=n)

    def emotional_search(self, target_vad: VAD, threshold: float = 0.75, limit: int = 10) -> list[tuple[Memory, float]]:
        """Find memories with similar emotional signatures."""
        return self.graph.find_by_emotion(target_vad, threshold, limit)

    def get_neighbors(self, node_id: str, edge_types: list[EdgeType] | None = None) -> list[tuple[Memory, Edge]]:
        return self.graph.get_neighbors(node_id, edge_types)

    def touch(self, memory_id: str):
        """Mark memory as accessed, strengthening its activation."""
        mem = self.graph.get_memory(memory_id)
        if mem:
            mem.touch()
            self.graph.update_memory(mem)

    def delete(self, memory_id: str):
        self.graph.delete_memory(memory_id)
        self.vector.delete(memory_id)

    def save(self):
        """Persist graph to disk. Vector store auto-persists via ChromaDB."""
        self.graph.save()

    def stats(self) -> dict:
        graph_stats = self.graph.stats()
        graph_stats["vector_count"] = self.vector.count()
        return graph_stats
