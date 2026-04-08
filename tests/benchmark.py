"""Benchmark: Standard RAG vs EAAM (Emotion-Anchored Associative Memory).

This benchmark measures whether emotion-anchored associative retrieval
outperforms flat vector search across scenarios designed to test
different dimensions of human-like memory retrieval.

Scoring:
- Each scenario has a query + expected relevant memories (ground truth)
- Both systems retrieve top-k results
- Precision@k, Recall@k, and NDCG@k are computed
- Additionally, an "Associative Leap Score" measures how well the system
  finds cross-domain connections that only emotional anchoring would surface.

Run: python3 tests/benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.consolidator import ConsolidationEngine
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.models import VAD, RetrievalResult
from eaam.store.memory_store import MemoryStore


# ============================================================================
# BENCHMARK SCENARIOS
# ============================================================================

@dataclass
class MemoryFixture:
    """A memory to populate the store with."""
    content: str
    conversation_id: str = ""
    topic: str = ""
    role: str = "user"
    tags: list[str] = field(default_factory=list)  # for ground truth matching


@dataclass
class QueryScenario:
    """A query with ground truth expected results."""
    name: str
    description: str
    query: str
    emotional_context: str  # the current emotional tone
    expected_tags: list[str]  # tags of memories that SHOULD be retrieved
    anti_tags: list[str] = field(default_factory=list)  # tags that should NOT appear
    category: str = "general"  # scenario type for grouped scoring


# --- The Memory Corpus ---
# 20 diverse memories across different domains and emotional registers

MEMORY_CORPUS = [
    # === Cluster 1: Negative tech incidents ===
    MemoryFixture(
        content="The production database crashed at 3am on a Saturday. I was panicking, hands shaking, couldn't figure out the root cause for two hours.",
        conversation_id="conv_incident1", topic="incident", tags=["neg_tech", "panic", "incident"],
    ),
    MemoryFixture(
        content="We lost customer data during the migration. The team was terrified about the fallout. Legal got involved immediately.",
        conversation_id="conv_incident2", topic="data_loss", tags=["neg_tech", "fear", "data_loss"],
    ),
    MemoryFixture(
        content="The load balancer failed during Black Friday traffic. I felt completely helpless watching error rates spike to 90%.",
        conversation_id="conv_incident3", topic="incident", tags=["neg_tech", "helpless", "outage"],
    ),

    # === Cluster 2: Positive tech wins ===
    MemoryFixture(
        content="Finally shipped the new recommendation engine. The whole team celebrated. Accuracy improved by 40% over the baseline.",
        conversation_id="conv_win1", topic="launch", tags=["pos_tech", "joy", "shipping"],
    ),
    MemoryFixture(
        content="Got the CI/CD pipeline working perfectly. Deployments went from 2 hours to 5 minutes. I was thrilled.",
        conversation_id="conv_win2", topic="devops", tags=["pos_tech", "joy", "devops"],
    ),
    MemoryFixture(
        content="The performance optimization reduced API latency from 800ms to 50ms. The client was ecstatic, sent a thank-you email.",
        conversation_id="conv_win3", topic="performance", tags=["pos_tech", "joy", "perf_win"],
    ),

    # === Cluster 3: Personal fear/anxiety (non-tech) ===
    MemoryFixture(
        content="I'm terrified about the upcoming job interview at Google. The anxiety keeps me up at night.",
        conversation_id="conv_personal1", topic="career", tags=["neg_personal", "fear", "interview"],
    ),
    MemoryFixture(
        content="The doctor said the test results were concerning and I need to come back for more tests. I've been scared all week.",
        conversation_id="conv_personal2", topic="health", tags=["neg_personal", "fear", "health"],
    ),

    # === Cluster 4: Personal joy (non-tech) ===
    MemoryFixture(
        content="My daughter took her first steps today. I cried with happiness. Best moment of my life.",
        conversation_id="conv_personal3", topic="family", tags=["pos_personal", "joy", "family"],
    ),
    MemoryFixture(
        content="Got accepted into the marathon I've been training for. Six months of preparation paid off, feeling on top of the world.",
        conversation_id="conv_personal4", topic="fitness", tags=["pos_personal", "joy", "achievement"],
    ),

    # === Cluster 5: Frustration/anger ===
    MemoryFixture(
        content="The vendor keeps missing deadlines and making excuses. I'm furious. Three months of delays on a critical integration.",
        conversation_id="conv_frust1", topic="vendor", tags=["anger", "frustration", "vendor"],
    ),
    MemoryFixture(
        content="Management rejected the refactoring proposal again despite the mounting tech debt. I'm angry and frustrated.",
        conversation_id="conv_frust2", topic="management", tags=["anger", "frustration", "mgmt"],
    ),
    MemoryFixture(
        content="The customer support tickets are piling up because of that broken feature. Users are furious and so am I.",
        conversation_id="conv_frust3", topic="support", tags=["anger", "frustration", "users"],
    ),

    # === Cluster 6: Calm/neutral technical ===
    MemoryFixture(
        content="Reviewed the architecture RFC for the new microservice. The design looks solid. Need to discuss caching strategy in the next meeting.",
        conversation_id="conv_neutral1", topic="architecture", tags=["neutral", "architecture"],
    ),
    MemoryFixture(
        content="Updated the database schema to add the new analytics columns. Ran the migration successfully on staging.",
        conversation_id="conv_neutral2", topic="database", tags=["neutral", "database"],
    ),

    # === Cluster 7: Surprise/discovery ===
    MemoryFixture(
        content="Found a critical security vulnerability in our auth system that's been there for two years. Nobody noticed until the audit.",
        conversation_id="conv_surprise1", topic="security", tags=["surprise", "security", "neg_tech"],
    ),
    MemoryFixture(
        content="Discovered that switching from JSON to protobuf cut our bandwidth by 70%. Completely unexpected improvement.",
        conversation_id="conv_surprise2", topic="optimization", tags=["surprise", "pos_tech", "discovery"],
    ),

    # === Cluster 8: Sadness/loss ===
    MemoryFixture(
        content="Our lead engineer resigned today. She was the backbone of the team. Everyone is devastated.",
        conversation_id="conv_sad1", topic="team", tags=["sadness", "loss", "team"],
    ),
    MemoryFixture(
        content="The startup we invested two years in is shutting down. All that work, all those late nights, gone.",
        conversation_id="conv_sad2", topic="startup", tags=["sadness", "loss", "startup"],
    ),

    # === Cluster 9: Mixed/complex ===
    MemoryFixture(
        content="Got promoted to principal engineer but it means relocating away from family. Bittersweet doesn't cover it.",
        conversation_id="conv_mixed1", topic="career", tags=["mixed", "career", "bittersweet"],
    ),
]


# --- The Query Scenarios ---
# Each tests a specific retrieval capability

QUERY_SCENARIOS = [
    # === Category: SEMANTIC BASELINE ===
    # These should be easy for both systems
    QueryScenario(
        name="direct_semantic",
        description="Direct semantic match — both systems should find this",
        query="database migration problems",
        emotional_context="neutral",
        expected_tags=["database", "data_loss"],
        category="semantic_baseline",
    ),
    QueryScenario(
        name="topic_match",
        description="Topic-based retrieval",
        query="production outage incident",
        emotional_context="neutral",
        expected_tags=["incident", "outage"],
        category="semantic_baseline",
    ),

    # === Category: EMOTIONAL CONGRUENCE ===
    # Same query, different emotion — EAAM should differentiate, RAG should not
    QueryScenario(
        name="emotional_shift_negative",
        description="Tech query with frustrated emotional context",
        query="dealing with technical challenges",
        emotional_context="I'm frustrated and angry about recurring problems",
        expected_tags=["anger", "frustration", "neg_tech"],
        anti_tags=["pos_tech", "joy"],
        category="emotional_congruence",
    ),
    QueryScenario(
        name="emotional_shift_positive",
        description="Same-ish query with happy emotional context",
        query="dealing with technical challenges",
        emotional_context="excited and proud of overcoming obstacles",
        expected_tags=["pos_tech", "joy", "achievement"],
        anti_tags=["anger", "frustration"],
        category="emotional_congruence",
    ),

    # === Category: CROSS-DOMAIN EMOTIONAL LEAPS ===
    # The key differentiator — finding emotionally similar but semantically distant memories
    QueryScenario(
        name="fear_crossdomain",
        description="Query about tech fear should also surface personal fears via emotional link",
        query="system security concerns keeping me up at night",
        emotional_context="anxious and worried, can't sleep",
        expected_tags=["fear", "security"],
        category="cross_domain_leap",
    ),
    QueryScenario(
        name="joy_crossdomain",
        description="Query about achievement should surface personal joys too",
        query="celebrating successful project completion",
        emotional_context="elated and proud",
        expected_tags=["joy", "shipping", "achievement"],
        category="cross_domain_leap",
    ),
    QueryScenario(
        name="loss_crossdomain",
        description="Loss in one domain should connect to loss in another",
        query="dealing with the loss of key team member",
        emotional_context="devastated and hopeless",
        expected_tags=["sadness", "loss"],
        category="cross_domain_leap",
    ),

    # === Category: MOOD-CONGRUENT RECALL ===
    # Vague query where emotional context should determine what surfaces
    QueryScenario(
        name="mood_negative",
        description="Vague query in negative mood should pull negative memories",
        query="thinking about work lately",
        emotional_context="depressed and overwhelmed",
        expected_tags=["neg_tech", "sadness", "frustration", "fear"],
        anti_tags=["pos_tech", "joy"],
        category="mood_congruent",
    ),
    QueryScenario(
        name="mood_positive",
        description="Same vague query in positive mood should pull positive memories",
        query="thinking about work lately",
        emotional_context="happy and grateful for my career",
        expected_tags=["pos_tech", "joy", "pos_personal", "achievement"],
        anti_tags=["neg_tech", "sadness", "anger"],
        category="mood_congruent",
    ),

    # === Category: ACTIVATION DYNAMICS ===
    # Tests whether frequently accessed or high-arousal memories get priority
    QueryScenario(
        name="high_arousal_priority",
        description="High-arousal memories should surface more easily with weak cues",
        query="something important happened recently",
        emotional_context="alert and concerned",
        expected_tags=["panic", "fear", "incident"],
        category="activation",
    ),
]


# ============================================================================
# BASELINE RAG RETRIEVER (vector-only, no emotional layer)
# ============================================================================

class BaselineRAGRetriever:
    """Simple vector-similarity-only retriever. No emotions, no graph, no spreading."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self, query: str, k: int = 5, **kwargs) -> list[RetrievalResult]:
        """Pure semantic retrieval — cosine similarity only."""
        hits = self.store.semantic_search(query, n=k)
        results = []
        for mem_id, sim_score in hits:
            memory = self.store.get(mem_id)
            if memory:
                results.append(RetrievalResult(
                    memory=memory,
                    score=sim_score,
                    semantic_score=sim_score,
                ))
        return results


# ============================================================================
# SCORING METRICS
# ============================================================================

def precision_at_k(retrieved_tags: list[set[str]], expected_tags: list[str], k: int) -> float:
    """What fraction of retrieved results are relevant?"""
    relevant = 0
    expected_set = set(expected_tags)
    for tags in retrieved_tags[:k]:
        if tags & expected_set:
            relevant += 1
    return relevant / k if k > 0 else 0.0


def recall_at_k(retrieved_tags: list[set[str]], expected_tags: list[str], k: int) -> float:
    """What fraction of expected tags appeared in results?"""
    expected_set = set(expected_tags)
    found = set()
    for tags in retrieved_tags[:k]:
        found.update(tags & expected_set)
    return len(found) / len(expected_set) if expected_set else 0.0


def anti_tag_score(retrieved_tags: list[set[str]], anti_tags: list[str], k: int) -> float:
    """Penalty: what fraction of results contain unwanted tags? (lower is better)"""
    if not anti_tags:
        return 0.0
    anti_set = set(anti_tags)
    violations = 0
    for tags in retrieved_tags[:k]:
        if tags & anti_set:
            violations += 1
    return violations / k if k > 0 else 0.0


def ndcg_at_k(retrieved_tags: list[set[str]], expected_tags: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain — rewards relevant results ranked higher."""
    expected_set = set(expected_tags)

    def dcg(tag_list: list[set[str]], n: int) -> float:
        score = 0.0
        for i, tags in enumerate(tag_list[:n]):
            rel = 1.0 if (tags & expected_set) else 0.0
            score += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        return score

    actual_dcg = dcg(retrieved_tags, k)
    # Ideal: all relevant first
    ideal_count = min(k, sum(1 for tags in retrieved_tags if tags & expected_set))
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(max(ideal_count, 1)))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ============================================================================
# MCP SIMULATION
# ============================================================================

def simulate_mcp_interaction(pipeline: EncodingPipeline, retriever: AssociativeRetriever):
    """Simulate a multi-turn MCP conversation to test memory building over time."""
    print("\n" + "=" * 70)
    print("MCP SIMULATION: Multi-turn conversation with memory building")
    print("=" * 70)

    turns = [
        ("conv_mcp", "user", "I've been debugging this auth issue for 3 hours and I'm losing my mind", "frustrated"),
        ("conv_mcp", "assistant", "I can see you're frustrated. Let me help trace through the authentication flow systematically.", "supportive"),
        ("conv_mcp", "user", "Wait, I think I found it — the JWT token is expired but the refresh logic has a race condition!", "surprised"),
        ("conv_mcp", "assistant", "That's a great catch. Race conditions in token refresh are a common but tricky issue.", "neutral"),
        ("conv_mcp", "user", "YES! Fixed it! The fix was just 3 lines. I can't believe I spent 3 hours on this but I'm so relieved now.", "relieved_happy"),
        ("conv_mcp", "user", "You know what, this reminds me of that time the database crashed at 3am. Same kind of panic turning to relief.", "reflective"),
    ]

    print("\n--- Encoding conversation turns ---")
    for conv_id, role, content, emotion_label in turns:
        mem = pipeline.encode(content=content, conversation_id=conv_id, role=role)
        print(f"  [{role}] {content[:60]}...")
        print(f"    -> V={mem.emotion.valence:.2f} A={mem.emotion.arousal:.2f} D={mem.emotion.dominance:.2f} | edges={len(pipeline.store.graph.get_outgoing_edges(mem.id))}")

    # Now test retrieval with different emotional contexts
    print("\n--- Retrieval tests after conversation ---")

    queries = [
        ("authentication problems", "frustrated and stuck", "Should find the debugging turns + past incidents"),
        ("authentication problems", "curious and learning", "Should shift toward the discovery/fix turns"),
        ("feeling relieved after fixing a hard bug", None, "Should find the relief moment + past relief memories"),
    ]

    for query, emo_ctx, expected_behavior in queries:
        results = retriever.retrieve(query, k=3, emotional_context=emo_ctx)
        print(f"\n  Query: \"{query}\" (emotion: {emo_ctx or 'from query'})")
        print(f"  Expected: {expected_behavior}")
        for i, r in enumerate(results):
            print(f"    [{i+1}] [{r.score:.3f}] {r.memory.content[:70]}...")
            print(f"        sem={r.semantic_score:.3f} emo={r.emotional_score:.3f} spread={r.spreading_score:.3f}")


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark():
    """Run the full benchmark comparing RAG vs EAAM."""
    use_transformer = "--transformer" in sys.argv or "-t" in sys.argv

    print("=" * 70)
    print("EAAM v2 BENCHMARK: Standard RAG vs Multi-Pathway Associative Memory")
    print(f"Emotion model: {'TRANSFORMER (accurate)' if use_transformer else 'LEXICON (fast)'}")
    print("=" * 70)
    print(f"Memory corpus: {len(MEMORY_CORPUS)} memories")
    print(f"Query scenarios: {len(QUERY_SCENARIOS)} queries")
    print()

    config = EAAMConfig()
    config.emotion.use_transformer = use_transformer
    config.graph.persist_path = tempfile.mkdtemp()
    config.vector.persist_path = tempfile.mkdtemp()

    store = MemoryStore(config)
    emotion_encoder = EmotionEncoder(config.emotion)
    pipeline = EncodingPipeline(store, emotion_encoder, config)
    eaam_retriever = AssociativeRetriever(store, emotion_encoder, config.retrieval)
    rag_retriever = BaselineRAGRetriever(store)

    # Build tag index for scoring
    tag_index: dict[str, set[str]] = {}  # memory_id -> set of tags

    # Populate the memory store
    print("--- Populating memory store ---")
    for fixture in MEMORY_CORPUS:
        mem = pipeline.encode(
            content=fixture.content,
            conversation_id=fixture.conversation_id,
            topic=fixture.topic,
            role=fixture.role,
        )
        tag_index[mem.id] = set(fixture.tags)
        print(f"  [{', '.join(fixture.tags)[:25]:25}] V={mem.emotion.valence:+.2f} A={mem.emotion.arousal:.2f} D={mem.emotion.dominance:.2f}")

    print(f"\nStore stats: {json.dumps(store.stats(), indent=2)}")

    # Run scenarios
    K = 5
    rag_scores: dict[str, list[dict]] = {}
    eaam_scores: dict[str, list[dict]] = {}

    print("\n" + "=" * 70)
    print("SCENARIO RESULTS")
    print("=" * 70)

    for scenario in QUERY_SCENARIOS:
        # RAG retrieval
        rag_results = rag_retriever.retrieve(scenario.query, k=K)
        rag_tags = [tag_index.get(r.memory.id, set()) for r in rag_results]

        # EAAM retrieval
        eaam_results = eaam_retriever.retrieve(
            scenario.query, k=K, emotional_context=scenario.emotional_context,
        )
        eaam_tags = [tag_index.get(r.memory.id, set()) for r in eaam_results]

        # Score both
        rag_metrics = {
            "precision": precision_at_k(rag_tags, scenario.expected_tags, K),
            "recall": recall_at_k(rag_tags, scenario.expected_tags, K),
            "ndcg": ndcg_at_k(rag_tags, scenario.expected_tags, K),
            "anti_violations": anti_tag_score(rag_tags, scenario.anti_tags, K),
        }
        eaam_metrics = {
            "precision": precision_at_k(eaam_tags, scenario.expected_tags, K),
            "recall": recall_at_k(eaam_tags, scenario.expected_tags, K),
            "ndcg": ndcg_at_k(eaam_tags, scenario.expected_tags, K),
            "anti_violations": anti_tag_score(eaam_tags, scenario.anti_tags, K),
        }

        # Composite: higher precision/recall/ndcg is better, lower anti_violations is better
        rag_composite = (rag_metrics["precision"] + rag_metrics["recall"] + rag_metrics["ndcg"]) / 3 - rag_metrics["anti_violations"] * 0.5
        eaam_composite = (eaam_metrics["precision"] + eaam_metrics["recall"] + eaam_metrics["ndcg"]) / 3 - eaam_metrics["anti_violations"] * 0.5

        cat = scenario.category
        rag_scores.setdefault(cat, []).append({**rag_metrics, "composite": rag_composite})
        eaam_scores.setdefault(cat, []).append({**eaam_metrics, "composite": eaam_composite})

        winner = "EAAM" if eaam_composite > rag_composite else ("RAG" if rag_composite > eaam_composite else "TIE")
        delta = eaam_composite - rag_composite

        print(f"\n[{scenario.category}] {scenario.name}")
        print(f"  Query: \"{scenario.query}\"")
        print(f"  Emotion: \"{scenario.emotional_context}\"")
        print(f"  Expected: {scenario.expected_tags}")
        print(f"  {'Metric':<12} {'RAG':>8} {'EAAM':>8} {'Delta':>8}")
        print(f"  {'─'*40}")
        print(f"  {'Precision':<12} {rag_metrics['precision']:>8.3f} {eaam_metrics['precision']:>8.3f} {eaam_metrics['precision']-rag_metrics['precision']:>+8.3f}")
        print(f"  {'Recall':<12} {rag_metrics['recall']:>8.3f} {eaam_metrics['recall']:>8.3f} {eaam_metrics['recall']-rag_metrics['recall']:>+8.3f}")
        print(f"  {'NDCG':<12} {rag_metrics['ndcg']:>8.3f} {eaam_metrics['ndcg']:>8.3f} {eaam_metrics['ndcg']-rag_metrics['ndcg']:>+8.3f}")
        print(f"  {'Anti-viol':<12} {rag_metrics['anti_violations']:>8.3f} {eaam_metrics['anti_violations']:>8.3f} {eaam_metrics['anti_violations']-rag_metrics['anti_violations']:>+8.3f}")
        print(f"  {'COMPOSITE':<12} {rag_composite:>8.3f} {eaam_composite:>8.3f} {delta:>+8.3f}  -> {winner}")

        # Show actual retrieved content
        print(f"\n  RAG top-3:")
        for i, r in enumerate(rag_results[:3]):
            tags_str = ", ".join(tag_index.get(r.memory.id, {"?"}))
            print(f"    [{i+1}] [{r.score:.3f}] ({tags_str}) {r.memory.content[:60]}...")
        print(f"  EAAM top-3:")
        for i, r in enumerate(eaam_results[:3]):
            tags_str = ", ".join(tag_index.get(r.memory.id, {"?"}))
            # Determine which pathway surfaced this memory
            pathway = "?"
            if r.semantic_score > 0: pathway = "hippocampal"
            elif r.emotional_score > 0: pathway = "amygdalar"
            elif r.spreading_score > 0: pathway = "spreading"
            elif r.activation_score > 0: pathway = "involuntary"
            print(f"    [{i+1}] [{r.score:.3f}] ({tags_str}) [{pathway}] {r.memory.content[:55]}...")

    # ======================================================================
    # AGGREGATE SCORES
    # ======================================================================
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS BY CATEGORY")
    print("=" * 70)

    all_categories = sorted(set(list(rag_scores.keys()) + list(eaam_scores.keys())))
    total_rag = 0.0
    total_eaam = 0.0
    total_n = 0

    for cat in all_categories:
        r_scores = rag_scores.get(cat, [])
        e_scores = eaam_scores.get(cat, [])
        n = len(r_scores)

        avg_rag = sum(s["composite"] for s in r_scores) / n if n else 0
        avg_eaam = sum(s["composite"] for s in e_scores) / n if n else 0
        delta = avg_eaam - avg_rag
        winner = "EAAM" if delta > 0.01 else ("RAG" if delta < -0.01 else "TIE")

        total_rag += sum(s["composite"] for s in r_scores)
        total_eaam += sum(s["composite"] for s in e_scores)
        total_n += n

        print(f"\n  {cat} ({n} scenarios)")
        print(f"    RAG avg composite:  {avg_rag:.3f}")
        print(f"    EAAM avg composite: {avg_eaam:.3f}")
        print(f"    Delta:              {delta:+.3f}  -> {winner}")

    print(f"\n{'=' * 70}")
    print(f"OVERALL RESULTS")
    print(f"{'=' * 70}")
    overall_rag = total_rag / total_n if total_n else 0
    overall_eaam = total_eaam / total_n if total_n else 0
    overall_delta = overall_eaam - overall_rag

    print(f"  RAG overall avg:     {overall_rag:.4f}")
    print(f"  EAAM overall avg:    {overall_eaam:.4f}")
    print(f"  Delta:               {overall_delta:+.4f}")
    print(f"  Winner:              {'EAAM' if overall_delta > 0 else 'RAG'}")
    print(f"  Improvement:         {abs(overall_delta / overall_rag) * 100:.1f}%" if overall_rag != 0 else "  N/A")

    # Expected winners by category
    print(f"\n  Expected category advantages:")
    print(f"    semantic_baseline:    RAG ~= EAAM (both should work)")
    print(f"    emotional_congruence: EAAM >> RAG (emotion shifts results)")
    print(f"    cross_domain_leap:    EAAM >> RAG (emotional associations)")
    print(f"    mood_congruent:       EAAM >> RAG (vague query + mood)")
    print(f"    activation:           EAAM > RAG  (arousal-modulated)")

    # ======================================================================
    # MCP SIMULATION
    # ======================================================================
    simulate_mcp_interaction(pipeline, eaam_retriever)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
