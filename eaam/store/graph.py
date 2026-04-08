"""In-memory graph store with JSON persistence.

Stores Memory nodes and weighted Edges. Supports neighbor traversal
and basic graph operations needed for spreading activation.
No external dependencies — works out of the box.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from eaam.config import GraphConfig
from eaam.models import Edge, EdgeType, Memory, VAD

logger = logging.getLogger(__name__)


class GraphStore:
    """In-memory graph with JSON persistence."""

    def __init__(self, config: GraphConfig | None = None):
        self.config = config or GraphConfig()
        self._nodes: dict[str, Memory] = {}
        self._edges: list[Edge] = []
        # Adjacency index: node_id -> list of (edge_index, direction)
        self._outgoing: dict[str, list[int]] = defaultdict(list)
        self._incoming: dict[str, list[int]] = defaultdict(list)
        self._load()

    # --- Node operations ---

    def add_memory(self, memory: Memory):
        self._nodes[memory.id] = memory

    def get_memory(self, memory_id: str) -> Memory | None:
        return self._nodes.get(memory_id)

    def get_all_memories(self) -> list[Memory]:
        return list(self._nodes.values())

    def update_memory(self, memory: Memory):
        self._nodes[memory.id] = memory

    def delete_memory(self, memory_id: str):
        self._nodes.pop(memory_id, None)
        # Remove associated edges
        self._edges = [e for e in self._edges if e.source_id != memory_id and e.target_id != memory_id]
        self._rebuild_adjacency()

    def count(self) -> int:
        return len(self._nodes)

    # --- Edge operations ---

    def add_edge(self, edge: Edge):
        idx = len(self._edges)
        self._edges.append(edge)
        self._outgoing[edge.source_id].append(idx)
        self._incoming[edge.target_id].append(idx)

    def get_outgoing_edges(self, node_id: str, edge_type: EdgeType | None = None) -> list[Edge]:
        edges = [self._edges[i] for i in self._outgoing.get(node_id, [])]
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_incoming_edges(self, node_id: str, edge_type: EdgeType | None = None) -> list[Edge]:
        edges = [self._edges[i] for i in self._incoming.get(node_id, [])]
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges

    def get_neighbors(self, node_id: str, edge_types: list[EdgeType] | None = None) -> list[tuple[Memory, Edge]]:
        """Get all neighbor memories with their connecting edges."""
        result = []
        for idx in self._outgoing.get(node_id, []):
            edge = self._edges[idx]
            if edge_types and edge.edge_type not in edge_types:
                continue
            neighbor = self._nodes.get(edge.target_id)
            if neighbor:
                result.append((neighbor, edge))

        # Also traverse incoming edges (graph is treated as undirected for retrieval)
        for idx in self._incoming.get(node_id, []):
            edge = self._edges[idx]
            if edge_types and edge.edge_type not in edge_types:
                continue
            neighbor = self._nodes.get(edge.source_id)
            if neighbor and neighbor.id != node_id:
                result.append((neighbor, edge))

        return result

    def get_all_edges(self) -> list[Edge]:
        return list(self._edges)

    def delete_edges_below(self, threshold: float):
        """Prune edges with weight below threshold."""
        before = len(self._edges)
        self._edges = [e for e in self._edges if e.weight >= threshold]
        self._rebuild_adjacency()
        pruned = before - len(self._edges)
        if pruned:
            logger.info("Pruned %d weak edges (threshold=%.2f)", pruned, threshold)

    def edge_exists(self, source_id: str, target_id: str, edge_type: EdgeType) -> bool:
        for idx in self._outgoing.get(source_id, []):
            e = self._edges[idx]
            if e.target_id == target_id and e.edge_type == edge_type:
                return True
        return False

    # --- Queries ---

    def find_by_emotion(self, target_vad: VAD, threshold: float = 0.75, limit: int = 10) -> list[tuple[Memory, float]]:
        """Find memories with similar emotional signatures."""
        results = []
        for mem in self._nodes.values():
            sim = target_vad.similarity(mem.emotion)
            if sim >= threshold:
                results.append((mem, sim))
        results.sort(key=lambda x: -x[1])
        return results[:limit]

    def find_by_conversation(self, conversation_id: str) -> list[Memory]:
        return [m for m in self._nodes.values() if m.conversation_id == conversation_id]

    # --- Persistence ---

    def save(self):
        path = Path(self.config.persist_path)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": [m.to_dict() for m in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

        filepath = path / "graph.json"
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug("Graph saved: %d nodes, %d edges -> %s", len(self._nodes), len(self._edges), filepath)

    def _load(self):
        filepath = Path(self.config.persist_path) / "graph.json"
        if not filepath.exists():
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            for node_data in data.get("nodes", []):
                mem = Memory.from_dict(node_data)
                self._nodes[mem.id] = mem

            for edge_data in data.get("edges", []):
                edge = Edge.from_dict(edge_data)
                idx = len(self._edges)
                self._edges.append(edge)
                self._outgoing[edge.source_id].append(idx)
                self._incoming[edge.target_id].append(idx)

            logger.info("Graph loaded: %d nodes, %d edges from %s", len(self._nodes), len(self._edges), filepath)
        except Exception as e:
            logger.error("Failed to load graph: %s", e)

    def _rebuild_adjacency(self):
        self._outgoing = defaultdict(list)
        self._incoming = defaultdict(list)
        for idx, edge in enumerate(self._edges):
            self._outgoing[edge.source_id].append(idx)
            self._incoming[edge.target_id].append(idx)

    # --- Stats ---

    def stats(self) -> dict:
        edge_type_counts = defaultdict(int)
        for e in self._edges:
            edge_type_counts[e.edge_type.value] += 1

        avg_activation = 0.0
        if self._nodes:
            avg_activation = sum(m.effective_activation() for m in self._nodes.values()) / len(self._nodes)

        return {
            "node_count": len(self._nodes),
            "edge_count": len(self._edges),
            "edge_types": dict(edge_type_counts),
            "avg_activation": round(avg_activation, 3),
            "reflection_count": sum(1 for m in self._nodes.values() if m.is_reflection),
        }
