"""Multi-pathway associative retriever — v2.

Human brains don't retrieve memories through a single scoring function.
They have parallel, independent retrieval circuits:

1. HIPPOCAMPAL (deliberate recall) — semantic similarity, conscious search
2. AMYGDALAR (emotional priming) — current emotional state activates
   emotionally congruent memories regardless of topic
3. SPREADING ACTIVATION (associative chaining) — activated nodes spread
   energy through the graph, crossing domains via emotional edges
4. INVOLUNTARY (Proust effect) — high-activation emotional hubs randomly
   surface with even weak cues, completely out of context

Each pathway runs INDEPENDENTLY and produces its own candidate set.
Results are merged with deduplication, preserving which pathway(s)
surfaced each memory.

This is fundamentally different from v1's single composite score.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eaam.emotion.encoder import EmotionEncoder
from eaam.models import Memory, RetrievalResult, VAD
from eaam.store.memory_store import MemoryStore

if TYPE_CHECKING:
    from eaam.config import RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass
class PathwayResult:
    """A single memory surfaced by a specific retrieval pathway."""
    memory: Memory
    score: float
    pathway: str  # which pathway found this
    reason: str = ""  # human-readable explanation
    path: list[str] = field(default_factory=list)


class AssociativeRetriever:
    """Multi-pathway associative retriever mimicking human memory circuits."""

    def __init__(
        self,
        store: MemoryStore,
        emotion_encoder: EmotionEncoder,
        config: RetrievalConfig | None = None,
    ):
        from eaam.config import RetrievalConfig

        self.store = store
        self.emotion = emotion_encoder
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        emotional_context: str | None = None,
        override_emotion: VAD | None = None,
    ) -> list[RetrievalResult]:
        """Run all retrieval pathways in parallel and merge results.

        Each pathway independently surfaces candidates. The final
        result set preserves diversity across pathways — a memory
        found by the emotional pathway that semantics would never
        find still gets a slot.
        """
        # Determine current emotional state
        if override_emotion is not None:
            current_emotion = override_emotion
        elif emotional_context:
            current_emotion = self.emotion.encode(emotional_context)
        else:
            current_emotion = self.emotion.encode(query)

        # Run all pathways independently
        hippocampal = self._pathway_hippocampal(query, k=k)
        amygdalar = self._pathway_amygdalar(current_emotion, k=k)
        spreading = self._pathway_spreading(query, current_emotion, k=k)
        involuntary = self._pathway_involuntary(query, current_emotion, k=max(2, k // 2))

        # Merge with pathway-aware deduplication
        merged = self._merge_pathways(
            hippocampal, amygdalar, spreading, involuntary, k=k
        )

        # Touch retrieved memories (retrieval strengthens them)
        for r in merged:
            self.store.touch(r.memory.id)
            # Reconsolidation: update emotional associations on retrieval
            self._reconsolidate(r.memory, current_emotion)

        return merged

    # ================================================================
    # PATHWAY 1: HIPPOCAMPAL — deliberate semantic recall
    # ================================================================

    def _pathway_hippocampal(self, query: str, k: int = 5) -> list[PathwayResult]:
        """Pure semantic search — what you'd get from standard RAG."""
        hits = self.store.semantic_search(query, n=k)
        results = []
        for mem_id, sim_score in hits:
            memory = self.store.get(mem_id)
            if memory and sim_score > 0.1:
                results.append(PathwayResult(
                    memory=memory,
                    score=sim_score,
                    pathway="hippocampal",
                    reason=f"semantic similarity {sim_score:.3f}",
                ))
        return results

    # ================================================================
    # PATHWAY 2: AMYGDALAR — emotional priming
    # ================================================================

    def _pathway_amygdalar(self, current_emotion: VAD, k: int = 5) -> list[PathwayResult]:
        """Find memories with matching emotional signatures.

        This is the mood-congruent recall pathway. When you're anxious,
        anxious memories surface regardless of topic. When you're happy,
        happy memories surface.

        Key insight: this pathway IGNORES semantics entirely.
        """
        # Search by emotional similarity across the full graph
        emotional_matches = self.store.emotional_search(
            current_emotion, threshold=0.70, limit=k * 2,
        )

        results = []
        for memory, emo_sim in emotional_matches:
            # Weight by both emotional similarity AND activation level
            # High-arousal memories are easier to prime emotionally
            activation = memory.effective_activation()
            score = emo_sim * 0.7 + activation * 0.3

            results.append(PathwayResult(
                memory=memory,
                score=score,
                pathway="amygdalar",
                reason=f"emotional resonance {emo_sim:.3f} (V={memory.emotion.valence:+.2f} A={memory.emotion.arousal:.2f})",
            ))

        results.sort(key=lambda r: -r.score)
        return results[:k]

    # ================================================================
    # PATHWAY 3: SPREADING ACTIVATION — associative graph traversal
    # ================================================================

    def _pathway_spreading(self, query: str, current_emotion: VAD, k: int = 5) -> list[PathwayResult]:
        """Spread activation from semantic seeds through the association graph.

        Seeds from semantic search, then propagates through ALL edge types.
        Emotional edges enable cross-domain leaps: a memory about a server
        crash can activate a memory about a health scare through shared fear.

        Uses ACT-R style summation (not max) so multi-path convergence
        amplifies activation.
        """
        # Seed from semantic candidates
        seeds = self.store.semantic_search(query, n=self.config.candidate_pool)
        if not seeds:
            return []

        # Activation map: id -> (activation, pathway_chain)
        activated: dict[str, tuple[float, list[str]]] = {}
        for mem_id, sim in seeds:
            activated[mem_id] = (sim, [mem_id])

        # Propagate through the graph
        # Key tuning: aggressive decay + fan-out limit prevents saturation
        EMOTION_BOOST = 0.3
        HOP_DECAY = 0.30  # much steeper than v1's 0.6 — prevents 0.99 scores

        for hop in range(self.config.activation_hops):
            decay = HOP_DECAY ** (hop + 1)
            new_act: dict[str, tuple[float, list[str]]] = {}

            for node_id, (activation, path) in list(activated.items()):
                if activation < 0.03:
                    continue

                neighbors = self.store.get_neighbors(node_id)
                # Limit fan-out to top 4 edges per node
                neighbors.sort(key=lambda x: -x[1].weight)
                for neighbor, edge in neighbors[:4]:
                    if neighbor.id in set(path):
                        continue

                    spread = activation * edge.weight * decay

                    emo_sim = current_emotion.similarity(neighbor.emotion)
                    spread *= (1.0 + EMOTION_BOOST * emo_sim)

                    new_path = path + [neighbor.id]

                    existing_act, existing_path = new_act.get(neighbor.id, (0.0, []))
                    # Sum but hard-cap at 0.7 — spreading should never dominate
                    combined = min(existing_act + spread, 0.7)
                    best_path = new_path if spread > existing_act else existing_path
                    new_act[neighbor.id] = (combined, best_path)

            for nid, (act, path) in new_act.items():
                old_act, old_path = activated.get(nid, (0.0, []))
                combined = min(old_act + act, 0.7)
                activated[nid] = (combined, path if act > old_act else old_path)

        # Only return memories that were reached VIA spreading (not seeds)
        seed_ids = {mem_id for mem_id, _ in seeds}
        results = []
        for mem_id, (act, path) in activated.items():
            if len(path) <= 1:
                continue  # skip seeds — hippocampal already handles them
            memory = self.store.get(mem_id)
            if memory and act > 0.05:
                results.append(PathwayResult(
                    memory=memory,
                    score=act,
                    pathway="spreading",
                    reason=f"{len(path)-1}-hop chain via {path[0][:12]}...",
                    path=path,
                ))

        results.sort(key=lambda r: -r.score)
        return results[:k]

    # ================================================================
    # PATHWAY 4: INVOLUNTARY — Proust effect / spontaneous recall
    # ================================================================

    def _pathway_involuntary(self, query: str, current_emotion: VAD, k: int = 2) -> list[PathwayResult]:
        """Simulate involuntary memory — high-activation emotional hubs
        that surface with even weak cues.

        In humans, this happens when:
        1. A sensory cue (smell, sound) triggers the amygdala directly
        2. A high-activation emotional memory has such low retrieval
           threshold that almost any related cue can fire it
        3. Random spreading activation hits a hub node

        Implementation:
        - Find the highest-activation emotional memories in the store
        - Check if ANY weak connection exists to the current context
        - If so, surface them with a "involuntary" tag
        - These can be completely out of context
        """
        all_memories = self.store.graph.get_all_memories()
        if not all_memories:
            return []

        # Find high-activation emotional hubs
        # Sort by: arousal * activation (highly emotional + highly activated)
        hubs = sorted(
            all_memories,
            key=lambda m: m.emotion.arousal * m.effective_activation(),
            reverse=True,
        )

        results = []
        for memory in hubs[:k * 3]:
            if memory.is_reflection:
                continue

            # Check for ANY weak connection to current context:
            # 1. Emotional resonance with current state
            emo_match = current_emotion.similarity(memory.emotion)
            # 2. Any shared feature with query (even partial word overlap)
            query_words = set(query.lower().split())
            content_words = set(memory.content.lower().split())
            word_overlap = len(query_words & content_words) / max(len(query_words), 1)

            # The threshold is deliberately LOW — involuntary memories
            # surface with weak cues. The "strength" comes from the
            # memory's internal activation, not the cue match.
            cue_strength = max(emo_match * 0.6, word_overlap * 0.4)

            # Involuntary firing probability:
            # high activation + high arousal + any cue = fires
            fire_score = (
                memory.effective_activation() * 0.4
                + memory.emotion.arousal * 0.3
                + cue_strength * 0.3
            )

            if fire_score > 0.35:
                results.append(PathwayResult(
                    memory=memory,
                    score=fire_score,
                    pathway="involuntary",
                    reason=f"Proust: arousal={memory.emotion.arousal:.2f} activation={memory.effective_activation():.2f} cue={cue_strength:.2f}",
                ))

        results.sort(key=lambda r: -r.score)
        return results[:k]

    # ================================================================
    # MERGE — combine pathways with diversity preservation
    # ================================================================

    def _merge_pathways(
        self,
        hippocampal: list[PathwayResult],
        amygdalar: list[PathwayResult],
        spreading: list[PathwayResult],
        involuntary: list[PathwayResult],
        k: int = 5,
    ) -> list[RetrievalResult]:
        """Merge results from all pathways with diversity guarantees.

        Strategy: allocate slots proportionally, but guarantee at least
        1 slot for each active non-hippocampal pathway. This ensures
        emotional, spreading, and involuntary memories always get
        representation — mimicking how human brains can't suppress
        emotionally charged involuntary memories.

        Slot allocation (for k=5):
          - Hippocampal: 2 slots (deliberate recall)
          - Amygdalar:   1 slot guaranteed (emotional priming)
          - Spreading:   1 slot guaranteed (associative leap)
          - Involuntary: 1 slot guaranteed (Proust effect)
          - Remaining:   best-score from any pathway
        """
        seen: set[str] = set()
        merged: list[RetrievalResult] = []

        def add_from_pathway(candidates: list[PathwayResult], n: int, label: str):
            added = 0
            for pr in candidates:
                if pr.memory.id in seen:
                    continue
                if added >= n:
                    break
                seen.add(pr.memory.id)
                merged.append(RetrievalResult(
                    memory=pr.memory,
                    score=pr.score,
                    semantic_score=pr.score if pr.pathway == "hippocampal" else 0.0,
                    emotional_score=pr.score if pr.pathway == "amygdalar" else 0.0,
                    activation_score=pr.score if pr.pathway == "involuntary" else 0.0,
                    spreading_score=pr.score if pr.pathway == "spreading" else 0.0,
                    path=pr.path or [pr.memory.id],
                ))
                added += 1
            return added

        # Phase 1: Guaranteed slots for each pathway
        # Hippocampal gets the most — deliberate recall is still primary
        hippocampal_slots = max(1, k // 3)
        add_from_pathway(hippocampal, hippocampal_slots, "hippocampal")

        # Each non-semantic pathway gets at least 1 guaranteed slot
        if amygdalar:
            add_from_pathway(amygdalar, 1, "amygdalar")
        if spreading:
            add_from_pathway(spreading, 1, "spreading")
        if involuntary:
            add_from_pathway(involuntary, 1, "involuntary")

        # Phase 2: Fill remaining slots with best-scoring from any pathway
        remaining = k - len(merged)
        if remaining > 0:
            all_remaining = []
            for pr_list in [hippocampal, amygdalar, spreading, involuntary]:
                for pr in pr_list:
                    if pr.memory.id not in seen:
                        all_remaining.append(pr)
            all_remaining.sort(key=lambda pr: -pr.score)
            add_from_pathway(all_remaining, remaining, "overflow")

        # Sort final results by score for presentation
        merged.sort(key=lambda r: -r.score)
        return merged[:k]

    # ================================================================
    # RECONSOLIDATION — update associations on retrieval
    # ================================================================

    def _reconsolidate(self, memory: Memory, current_emotion: VAD):
        """When a memory is retrieved in a new emotional context,
        slightly shift its emotional signature toward the current state.

        This mirrors human reconsolidation — recalled memories absorb
        some of the current context, which is why memories change
        slightly each time we recall them.
        """
        RECONSOLIDATION_RATE = 0.05  # subtle shift per retrieval

        # Blend current emotion into the memory's signature
        memory.emotion.valence += RECONSOLIDATION_RATE * (current_emotion.valence - memory.emotion.valence)
        memory.emotion.arousal += RECONSOLIDATION_RATE * (current_emotion.arousal - memory.emotion.arousal)
        memory.emotion.dominance += RECONSOLIDATION_RATE * (current_emotion.dominance - memory.emotion.dominance)

        # Clamp (VAD.__post_init__ handles this but be explicit)
        memory.emotion.__post_init__()
        self.store.graph.update_memory(memory)

    # ================================================================
    # ASSOCIATIVE WALK — Proust mode (unchanged from v1)
    # ================================================================

    def associative_walk(
        self,
        start_memory_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Walk the association graph from a starting memory."""
        from eaam.models import EdgeType

        type_filter = None
        if edge_types:
            type_filter = [EdgeType(t) for t in edge_types]

        visited: set[str] = set()
        results: list[RetrievalResult] = []
        frontier = [(start_memory_id, 1.0, [start_memory_id])]

        for depth in range(max_depth):
            next_frontier = []
            for node_id, parent_activation, path in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                memory = self.store.get(node_id)
                if memory is None:
                    continue

                if node_id != start_memory_id:
                    results.append(RetrievalResult(
                        memory=memory,
                        score=parent_activation,
                        path=path,
                    ))

                neighbors = self.store.get_neighbors(node_id, type_filter)
                neighbors.sort(key=lambda x: -x[1].weight)
                for neighbor, edge in neighbors[:3]:
                    if neighbor.id not in visited:
                        child_activation = parent_activation * edge.weight * 0.7
                        next_frontier.append((
                            neighbor.id,
                            child_activation,
                            path + [neighbor.id],
                        ))

            frontier = next_frontier

        results.sort(key=lambda r: -r.score)
        return results
