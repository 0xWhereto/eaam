"""Consolidation engine — periodic background process for memory refinement.

Mirrors human sleep consolidation:
1. Decay: weaken unaccessed memories
2. Strengthen: boost high-arousal, frequently accessed memories
3. Cluster: find emotional communities in the graph
4. Abstract: generate reflection nodes from clusters
5. Prune: remove weak edges
6. Reconsolidate: update associations on re-accessed memories
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from eaam.models import Edge, EdgeType, Memory, VAD
from eaam.store.memory_store import MemoryStore

if TYPE_CHECKING:
    from eaam.config import ConsolidationConfig

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Periodic memory consolidation — the 'sleep' cycle."""

    def __init__(self, store: MemoryStore, config: ConsolidationConfig | None = None):
        from eaam.config import ConsolidationConfig

        self.store = store
        self.config = config or ConsolidationConfig()

    def run(self, generate_reflections: bool = True) -> dict:
        """Run a full consolidation cycle. Returns stats about what changed."""
        stats = {
            "decayed": 0,
            "strengthened": 0,
            "edges_pruned": 0,
            "clusters_found": 0,
            "reflections_created": 0,
        }

        stats["decayed"] = self._decay()
        stats["strengthened"] = self._strengthen()
        stats["edges_pruned"] = self._prune_edges()

        if generate_reflections:
            clusters = self._find_emotional_clusters()
            stats["clusters_found"] = len(clusters)
            stats["reflections_created"] = self._create_reflections(clusters)

        self.store.save()
        logger.info("Consolidation complete: %s", stats)
        return stats

    def _decay(self) -> int:
        """Reduce base activation of all memories (time-based decay)."""
        count = 0
        for memory in self.store.graph.get_all_memories():
            if memory.is_reflection:
                continue
            old = memory.base_activation
            memory.base_activation *= self.config.decay_rate
            memory.base_activation = max(memory.base_activation, 0.01)
            if memory.base_activation < old:
                count += 1
            self.store.graph.update_memory(memory)
        return count

    def _strengthen(self) -> int:
        """Boost high-arousal, frequently accessed memories."""
        count = 0
        for memory in self.store.graph.get_all_memories():
            if memory.is_reflection:
                continue
            # Strengthen if: high arousal AND accessed multiple times
            if memory.emotion.arousal >= self.config.strengthen_threshold and memory.access_count >= 2:
                memory.base_activation = min(memory.base_activation * 1.08, 1.0)
                self.store.graph.update_memory(memory)
                count += 1
        return count

    def _prune_edges(self) -> int:
        """Remove edges below weight threshold."""
        before = len(self.store.graph.get_all_edges())
        self.store.graph.delete_edges_below(self.config.prune_edge_threshold)
        after = len(self.store.graph.get_all_edges())
        return before - after

    def _find_emotional_clusters(self) -> list[list[Memory]]:
        """Find clusters of memories with similar emotional signatures.

        Uses a simple greedy clustering approach:
        - For each unvisited memory, find all emotionally similar neighbors
        - Group them into clusters of min_cluster_size or more
        """
        memories = [m for m in self.store.graph.get_all_memories() if not m.is_reflection]
        visited: set[str] = set()
        clusters: list[list[Memory]] = []

        for memory in memories:
            if memory.id in visited:
                continue

            # Find emotionally similar memories
            cluster = [memory]
            visited.add(memory.id)

            for other in memories:
                if other.id in visited:
                    continue
                emo_sim = memory.emotion.similarity(other.emotion)
                if emo_sim >= 0.80:
                    cluster.append(other)
                    visited.add(other.id)

            if len(cluster) >= self.config.min_cluster_size:
                clusters.append(cluster)

        return clusters

    def _create_reflections(self, clusters: list[list[Memory]]) -> int:
        """Create reflection nodes that synthesize emotional clusters.

        Each reflection is a higher-order memory that captures the
        emotional pattern across a cluster, linked back to source memories.
        """
        created = 0
        for cluster in clusters:
            # Compute average VAD for the cluster
            avg_v = sum(m.emotion.valence for m in cluster) / len(cluster)
            avg_a = sum(m.emotion.arousal for m in cluster) / len(cluster)
            avg_d = sum(m.emotion.dominance for m in cluster) / len(cluster)
            avg_vad = VAD(valence=avg_v, arousal=avg_a, dominance=avg_d)

            # Build a summary from the cluster's content
            summaries = [m.summary or m.content[:80] for m in cluster[:5]]
            content = f"Reflection: {len(cluster)} related memories share emotional pattern " \
                      f"(V={avg_v:.2f}, A={avg_a:.2f}, D={avg_d:.2f}). " \
                      f"Key threads: {'; '.join(summaries)}"

            reflection = Memory(
                content=content,
                summary=f"Emotional cluster ({len(cluster)} memories)",
                emotion=avg_vad,
                base_activation=0.6,  # reflections start with moderate activation
                is_reflection=True,
            )

            self.store.add(reflection)

            # Link reflection to source memories
            for source_mem in cluster:
                self.store.add_edge(Edge(
                    source_id=reflection.id,
                    target_id=source_mem.id,
                    edge_type=EdgeType.REFLECTION,
                    weight=0.8,
                ))

            created += 1

        return created

    def get_emotional_landscape(self) -> dict:
        """Analyze the emotional distribution of all stored memories."""
        memories = self.store.graph.get_all_memories()
        if not memories:
            return {"total": 0, "quadrants": {}, "avg_arousal": 0, "avg_dominance": 0}

        positive = [m for m in memories if m.emotion.valence > 0.2]
        negative = [m for m in memories if m.emotion.valence < -0.2]
        neutral_mems = [m for m in memories if -0.2 <= m.emotion.valence <= 0.2]

        high_arousal = [m for m in memories if m.emotion.arousal > 0.6]
        low_arousal = [m for m in memories if m.emotion.arousal <= 0.4]

        return {
            "total": len(memories),
            "reflections": sum(1 for m in memories if m.is_reflection),
            "emotional_distribution": {
                "positive": len(positive),
                "negative": len(negative),
                "neutral": len(neutral_mems),
            },
            "arousal_distribution": {
                "high": len(high_arousal),
                "low": len(low_arousal),
            },
            "avg_valence": sum(m.emotion.valence for m in memories) / len(memories),
            "avg_arousal": sum(m.emotion.arousal for m in memories) / len(memories),
            "avg_dominance": sum(m.emotion.dominance for m in memories) / len(memories),
            "avg_activation": sum(m.effective_activation() for m in memories) / len(memories),
            "top_activated": [
                {"id": m.id, "summary": m.summary, "activation": round(m.effective_activation(), 3)}
                for m in sorted(memories, key=lambda m: -m.effective_activation())[:5]
            ],
        }
