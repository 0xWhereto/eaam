"""EAAM Auto-Tuner: 100-iteration self-adjusting optimization loop.

Each iteration:
1. Run the full benchmark
2. Analyze which categories are weak
3. Adjust parameters to improve weak categories
4. If overall score declined, rollback and try a different adjustment
5. Track the best configuration seen

Tunable parameters:
- Retriever: HOP_DECAY, EMOTION_BOOST, spreading cap, fan-out limit
- Merge: hippocampal_slots ratio, amygdalar threshold
- Involuntary: fire threshold, arousal weight, cue weight
- Amygdalar: emo_sim threshold, activation blend
- Encoding: semantic edge threshold, emotional edge threshold, semantic_sim filter
"""

from __future__ import annotations

import copy
import json
import math
import random
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
from eaam.models import VAD, Memory, RetrievalResult
from eaam.store.memory_store import MemoryStore


# ============================================================================
# TUNABLE PARAMETERS (the genome we're optimizing)
# ============================================================================

@dataclass
class TunableParams:
    """All tunable parameters in one place for easy snapshot/restore."""
    # Spreading activation
    hop_decay: float = 0.30
    emotion_boost_spreading: float = 0.30
    spreading_cap: float = 0.70
    fan_out_limit: int = 4

    # Merge strategy
    hippocampal_ratio: float = 0.33  # fraction of k for hippocampal slots

    # Amygdalar pathway
    amygdalar_emo_threshold: float = 0.70
    amygdalar_emo_weight: float = 0.70
    amygdalar_act_weight: float = 0.30

    # Involuntary pathway
    involuntary_fire_threshold: float = 0.35
    involuntary_arousal_weight: float = 0.30
    involuntary_activation_weight: float = 0.40
    involuntary_cue_weight: float = 0.30

    # Encoding thresholds
    semantic_edge_threshold: float = 0.30
    emotional_edge_threshold: float = 0.75
    emotional_semantic_filter: float = 0.50  # only create emotional edges when sem < this

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def clone(self) -> TunableParams:
        return TunableParams(**self.to_dict())


# ============================================================================
# MEMORY CORPUS + SCENARIOS (imported from benchmark)
# ============================================================================

from tests.benchmark import (
    MEMORY_CORPUS, QUERY_SCENARIOS, BaselineRAGRetriever,
    precision_at_k, recall_at_k, ndcg_at_k, anti_tag_score,
)


# ============================================================================
# PATCHED COMPONENTS (apply tunable params at runtime)
# ============================================================================

def build_system(params: TunableParams, use_transformer: bool = True):
    """Build a fresh EAAM system with the given parameters."""
    config = EAAMConfig()
    config.emotion.use_transformer = use_transformer
    config.graph.persist_path = tempfile.mkdtemp()
    config.vector.persist_path = tempfile.mkdtemp()

    store = MemoryStore(config)
    emotion_encoder = EmotionEncoder(config.emotion)
    pipeline = EncodingPipeline(store, emotion_encoder, config)
    retriever = AssociativeRetriever(store, emotion_encoder, config.retrieval)
    rag = BaselineRAGRetriever(store)

    # Patch encoding thresholds
    pipeline._semantic_edge_threshold = params.semantic_edge_threshold
    pipeline._emotional_edge_threshold = params.emotional_edge_threshold
    pipeline._emotional_semantic_filter = params.emotional_semantic_filter

    # Monkey-patch the encoding methods to use tunable thresholds
    original_build_semantic = pipeline._build_semantic_edges
    def patched_semantic(memory):
        candidates = pipeline.store.semantic_search(memory.content, n=10)
        from eaam.models import Edge, EdgeType
        for mem_id, similarity in candidates:
            if mem_id == memory.id:
                continue
            if similarity < params.semantic_edge_threshold:
                continue
            pipeline.store.add_edge(Edge(
                source_id=memory.id, target_id=mem_id,
                edge_type=EdgeType.SEMANTIC, weight=similarity,
            ))
    pipeline._build_semantic_edges = patched_semantic

    original_build_emotional = pipeline._build_emotional_edges
    def patched_emotional(memory):
        emotional_neighbors = pipeline.store.emotional_search(
            memory.emotion, threshold=params.emotional_edge_threshold, limit=8
        )
        semantic_candidates = dict(pipeline.store.semantic_search(memory.content, n=20))
        from eaam.models import Edge, EdgeType
        for neighbor, emo_sim in emotional_neighbors:
            if neighbor.id == memory.id:
                continue
            semantic_sim = semantic_candidates.get(neighbor.id, 0.0)
            if semantic_sim < params.emotional_semantic_filter:
                pipeline.store.add_edge(Edge(
                    source_id=memory.id, target_id=neighbor.id,
                    edge_type=EdgeType.EMOTIONAL, weight=emo_sim,
                    metadata={"shared_valence": (memory.emotion.valence + neighbor.emotion.valence) / 2},
                ))
    pipeline._build_emotional_edges = patched_emotional

    return store, emotion_encoder, pipeline, retriever, rag, params


def run_retrieval_with_params(retriever: AssociativeRetriever, params: TunableParams,
                               query: str, k: int, emotional_context: str | None,
                               current_emotion: VAD) -> list[RetrievalResult]:
    """Run retrieval with patched parameters applied to the retriever."""
    from eaam.engine.retriever import PathwayResult

    # Pathway 1: Hippocampal (unchanged — pure semantic)
    hippocampal = retriever._pathway_hippocampal(query, k=k)

    # Pathway 2: Amygdalar (patched thresholds)
    emotional_matches = retriever.store.emotional_search(
        current_emotion, threshold=params.amygdalar_emo_threshold, limit=k * 2,
    )
    amygdalar = []
    for memory, emo_sim in emotional_matches:
        activation = memory.effective_activation()
        score = emo_sim * params.amygdalar_emo_weight + activation * params.amygdalar_act_weight
        amygdalar.append(PathwayResult(
            memory=memory, score=score, pathway="amygdalar",
            reason=f"emo={emo_sim:.3f}",
        ))
    amygdalar.sort(key=lambda r: -r.score)
    amygdalar = amygdalar[:k]

    # Pathway 3: Spreading (patched parameters)
    seeds = retriever.store.semantic_search(query, n=retriever.config.candidate_pool)
    activated: dict[str, tuple[float, list[str]]] = {}
    for mem_id, sim in (seeds or []):
        activated[mem_id] = (sim, [mem_id])

    for hop in range(retriever.config.activation_hops):
        decay = params.hop_decay ** (hop + 1)
        new_act: dict[str, tuple[float, list[str]]] = {}
        for node_id, (activation, path) in list(activated.items()):
            if activation < 0.03:
                continue
            neighbors = retriever.store.get_neighbors(node_id)
            neighbors.sort(key=lambda x: -x[1].weight)
            for neighbor, edge in neighbors[:params.fan_out_limit]:
                if neighbor.id in set(path):
                    continue
                spread = activation * edge.weight * decay
                emo_sim = current_emotion.similarity(neighbor.emotion)
                spread *= (1.0 + params.emotion_boost_spreading * emo_sim)
                new_path = path + [neighbor.id]
                existing_act, existing_path = new_act.get(neighbor.id, (0.0, []))
                combined = min(existing_act + spread, params.spreading_cap)
                best_path = new_path if spread > existing_act else existing_path
                new_act[neighbor.id] = (combined, best_path)
        for nid, (act, path) in new_act.items():
            old_act, old_path = activated.get(nid, (0.0, []))
            combined = min(old_act + act, params.spreading_cap)
            activated[nid] = (combined, path if act > old_act else old_path)

    spreading = []
    for mem_id, (act, path) in activated.items():
        if len(path) <= 1:
            continue
        memory = retriever.store.get(mem_id)
        if memory and act > 0.05:
            spreading.append(PathwayResult(
                memory=memory, score=act, pathway="spreading",
                reason=f"{len(path)-1}-hop", path=path,
            ))
    spreading.sort(key=lambda r: -r.score)
    spreading = spreading[:k]

    # Pathway 4: Involuntary
    all_memories = retriever.store.graph.get_all_memories()
    hubs = sorted(all_memories, key=lambda m: m.emotion.arousal * m.effective_activation(), reverse=True)
    involuntary = []
    for memory in hubs[:k * 3]:
        if memory.is_reflection:
            continue
        emo_match = current_emotion.similarity(memory.emotion)
        query_words = set(query.lower().split())
        content_words = set(memory.content.lower().split())
        word_overlap = len(query_words & content_words) / max(len(query_words), 1)
        cue_strength = max(emo_match * 0.6, word_overlap * 0.4)
        fire_score = (
            memory.effective_activation() * params.involuntary_activation_weight
            + memory.emotion.arousal * params.involuntary_arousal_weight
            + cue_strength * params.involuntary_cue_weight
        )
        if fire_score > params.involuntary_fire_threshold:
            involuntary.append(PathwayResult(
                memory=memory, score=fire_score, pathway="involuntary",
                reason=f"Proust: fire={fire_score:.3f}",
            ))
    involuntary.sort(key=lambda r: -r.score)
    involuntary = involuntary[:max(2, k // 2)]

    # Merge
    return _merge_with_params(hippocampal, amygdalar, spreading, involuntary, k, params)


def _merge_with_params(hippocampal, amygdalar, spreading, involuntary, k, params):
    """Merge with tunable hippocampal ratio."""
    seen: set[str] = set()
    merged: list[RetrievalResult] = []

    def add(candidates, n):
        added = 0
        for pr in candidates:
            if pr.memory.id in seen:
                continue
            if added >= n:
                break
            seen.add(pr.memory.id)
            merged.append(RetrievalResult(
                memory=pr.memory, score=pr.score,
                semantic_score=pr.score if pr.pathway == "hippocampal" else 0.0,
                emotional_score=pr.score if pr.pathway == "amygdalar" else 0.0,
                activation_score=pr.score if pr.pathway == "involuntary" else 0.0,
                spreading_score=pr.score if pr.pathway == "spreading" else 0.0,
                path=pr.path or [pr.memory.id],
            ))
            added += 1
        return added

    hip_slots = max(1, int(k * params.hippocampal_ratio))
    add(hippocampal, hip_slots)
    if amygdalar: add(amygdalar, 1)
    if spreading: add(spreading, 1)
    if involuntary: add(involuntary, 1)

    remaining = k - len(merged)
    if remaining > 0:
        all_rem = []
        for lst in [hippocampal, amygdalar, spreading, involuntary]:
            for pr in lst:
                if pr.memory.id not in seen:
                    all_rem.append(pr)
        all_rem.sort(key=lambda pr: -pr.score)
        add(all_rem, remaining)

    merged.sort(key=lambda r: -r.score)
    return merged[:k]


# ============================================================================
# BENCHMARK SCORER
# ============================================================================

def score_benchmark(store, emotion_encoder, pipeline, retriever, rag, params, tag_index) -> dict:
    """Run all scenarios and return category + overall scores."""
    K = 5
    cat_scores_rag: dict[str, list[float]] = {}
    cat_scores_eaam: dict[str, list[float]] = {}

    for scenario in QUERY_SCENARIOS:
        # RAG
        rag_results = rag.retrieve(scenario.query, k=K)
        rag_tags = [tag_index.get(r.memory.id, set()) for r in rag_results]

        # EAAM with patched params
        if scenario.emotional_context:
            current_emotion = emotion_encoder.encode(scenario.emotional_context)
        else:
            current_emotion = emotion_encoder.encode(scenario.query)

        eaam_results = run_retrieval_with_params(
            retriever, params, scenario.query, K, scenario.emotional_context, current_emotion
        )
        eaam_tags = [tag_index.get(r.memory.id, set()) for r in eaam_results]

        for tags_list, cat_dict in [(rag_tags, cat_scores_rag), (eaam_tags, cat_scores_eaam)]:
            p = precision_at_k(tags_list, scenario.expected_tags, K)
            r = recall_at_k(tags_list, scenario.expected_tags, K)
            n = ndcg_at_k(tags_list, scenario.expected_tags, K)
            a = anti_tag_score(tags_list, scenario.anti_tags, K)
            composite = (p + r + n) / 3 - a * 0.5
            cat_dict.setdefault(scenario.category, []).append(composite)

    # Aggregate
    result = {"categories": {}, "overall_rag": 0.0, "overall_eaam": 0.0}
    total_rag, total_eaam, total_n = 0.0, 0.0, 0
    for cat in sorted(set(list(cat_scores_rag.keys()) + list(cat_scores_eaam.keys()))):
        avg_rag = sum(cat_scores_rag.get(cat, [0])) / max(len(cat_scores_rag.get(cat, [1])), 1)
        avg_eaam = sum(cat_scores_eaam.get(cat, [0])) / max(len(cat_scores_eaam.get(cat, [1])), 1)
        result["categories"][cat] = {"rag": avg_rag, "eaam": avg_eaam, "delta": avg_eaam - avg_rag}
        n = len(cat_scores_rag.get(cat, []))
        total_rag += sum(cat_scores_rag.get(cat, []))
        total_eaam += sum(cat_scores_eaam.get(cat, []))
        total_n += n

    result["overall_rag"] = total_rag / max(total_n, 1)
    result["overall_eaam"] = total_eaam / max(total_n, 1)
    result["delta"] = result["overall_eaam"] - result["overall_rag"]
    result["improvement_pct"] = (result["delta"] / max(result["overall_rag"], 0.001)) * 100
    return result


# ============================================================================
# PARAMETER MUTATION STRATEGIES
# ============================================================================

def mutate_params(params: TunableParams, scores: dict, iteration: int) -> TunableParams:
    """Intelligently mutate parameters based on which categories are weak."""
    new = params.clone()
    cats = scores["categories"]

    # Identify weakest category
    weakest_cat = min(cats, key=lambda c: cats[c]["delta"])
    weakest_delta = cats[weakest_cat]["delta"]

    # Adaptive mutation magnitude: larger early, smaller later
    mag = max(0.02, 0.15 * (1.0 - iteration / 100))

    # Random exploration component (20% of the time, just try something random)
    if random.random() < 0.20:
        param_name = random.choice([
            "hop_decay", "emotion_boost_spreading", "spreading_cap", "fan_out_limit",
            "amygdalar_emo_threshold", "amygdalar_emo_weight", "amygdalar_act_weight",
            "involuntary_fire_threshold", "involuntary_arousal_weight",
            "involuntary_activation_weight", "involuntary_cue_weight",
            "hippocampal_ratio", "semantic_edge_threshold",
            "emotional_edge_threshold", "emotional_semantic_filter",
        ])
        current_val = getattr(new, param_name)
        if isinstance(current_val, int):
            delta = random.choice([-1, 1])
            setattr(new, param_name, max(1, min(8, current_val + delta)))
        else:
            delta = random.uniform(-mag, mag)
            setattr(new, param_name, max(0.01, min(0.99, current_val + delta)))
        return new

    # Targeted mutations based on weak category
    if weakest_cat == "semantic_baseline" and weakest_delta < -0.1:
        # Semantic is weak: give hippocampal more slots, reduce other pathways
        new.hippocampal_ratio = min(0.60, new.hippocampal_ratio + mag * 0.5)
        new.involuntary_fire_threshold = min(0.60, new.involuntary_fire_threshold + mag * 0.3)

    elif weakest_cat == "emotional_congruence" and weakest_delta < 0.3:
        # Emotional congruence is weak: boost amygdalar pathway
        new.amygdalar_emo_threshold = max(0.50, new.amygdalar_emo_threshold - mag * 0.5)
        new.amygdalar_emo_weight = min(0.90, new.amygdalar_emo_weight + mag * 0.3)

    elif weakest_cat == "cross_domain_leap" and weakest_delta < 0:
        # Cross-domain is weak: improve emotional edge building + spreading
        new.emotional_edge_threshold = max(0.55, new.emotional_edge_threshold - mag * 0.3)
        new.emotional_semantic_filter = min(0.70, new.emotional_semantic_filter + mag * 0.3)
        new.emotion_boost_spreading = min(0.60, new.emotion_boost_spreading + mag * 0.2)

    elif weakest_cat == "mood_congruent" and weakest_delta < 0.2:
        # Mood congruent is weak: emotional pathway needs more influence
        new.amygdalar_emo_weight = min(0.90, new.amygdalar_emo_weight + mag * 0.3)
        new.hippocampal_ratio = max(0.15, new.hippocampal_ratio - mag * 0.2)

    elif weakest_cat == "activation" and weakest_delta < 0:
        # Activation dynamics weak: boost involuntary pathway
        new.involuntary_fire_threshold = max(0.20, new.involuntary_fire_threshold - mag * 0.4)
        new.involuntary_activation_weight = min(0.60, new.involuntary_activation_weight + mag * 0.3)
        new.involuntary_arousal_weight = min(0.50, new.involuntary_arousal_weight + mag * 0.2)

    else:
        # General improvement: try small random tweaks to multiple params
        for attr in ["hop_decay", "spreading_cap", "emotion_boost_spreading"]:
            current = getattr(new, attr)
            setattr(new, attr, max(0.05, min(0.95, current + random.uniform(-mag * 0.5, mag * 0.5))))

    return new


# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

def run_optimization(n_iterations: int = 100, use_transformer: bool = True):
    print("=" * 70)
    print(f"EAAM AUTO-TUNER: {n_iterations} iterations")
    print(f"Emotion model: {'TRANSFORMER' if use_transformer else 'LEXICON'}")
    print("=" * 70)

    # Start with current best params
    current_params = TunableParams()
    best_params = current_params.clone()
    best_score = -999.0
    best_iteration = 0

    # History for progress tracking
    history: list[dict] = []
    rollback_count = 0
    improve_count = 0

    # Pre-build the emotion encoder once (slow to load)
    config = EAAMConfig()
    config.emotion.use_transformer = use_transformer
    shared_emotion_encoder = EmotionEncoder(config.emotion)

    for iteration in range(1, n_iterations + 1):
        iter_start = time.time()

        # Build fresh system with current params
        config = EAAMConfig()
        config.emotion.use_transformer = use_transformer
        config.graph.persist_path = tempfile.mkdtemp()
        config.vector.persist_path = tempfile.mkdtemp()

        store = MemoryStore(config)
        # Reuse the shared emotion encoder (avoid reloading model)
        pipeline = EncodingPipeline(store, shared_emotion_encoder, config)
        retriever = AssociativeRetriever(store, shared_emotion_encoder, config.retrieval)
        rag = BaselineRAGRetriever(store)

        # Patch encoding with current params
        from eaam.models import Edge, EdgeType
        def make_patched_semantic(p, pip):
            def patched(memory):
                candidates = pip.store.semantic_search(memory.content, n=10)
                for mem_id, similarity in candidates:
                    if mem_id == memory.id: continue
                    if similarity < p.semantic_edge_threshold: continue
                    pip.store.add_edge(Edge(source_id=memory.id, target_id=mem_id,
                                            edge_type=EdgeType.SEMANTIC, weight=similarity))
            return patched

        def make_patched_emotional(p, pip):
            def patched(memory):
                emotional_neighbors = pip.store.emotional_search(
                    memory.emotion, threshold=p.emotional_edge_threshold, limit=8)
                semantic_candidates = dict(pip.store.semantic_search(memory.content, n=20))
                for neighbor, emo_sim in emotional_neighbors:
                    if neighbor.id == memory.id: continue
                    semantic_sim = semantic_candidates.get(neighbor.id, 0.0)
                    if semantic_sim < p.emotional_semantic_filter:
                        pip.store.add_edge(Edge(source_id=memory.id, target_id=neighbor.id,
                                                edge_type=EdgeType.EMOTIONAL, weight=emo_sim))
            return patched

        pipeline._build_semantic_edges = make_patched_semantic(current_params, pipeline)
        pipeline._build_emotional_edges = make_patched_emotional(current_params, pipeline)

        # Populate memory store
        tag_index: dict[str, set[str]] = {}
        for fixture in MEMORY_CORPUS:
            mem = pipeline.encode(
                content=fixture.content,
                conversation_id=fixture.conversation_id,
                topic=fixture.topic, role=fixture.role,
            )
            tag_index[mem.id] = set(fixture.tags)

        # Score
        scores = score_benchmark(store, shared_emotion_encoder, pipeline, retriever, rag, current_params, tag_index)
        current_score = scores["overall_eaam"]
        delta_vs_rag = scores["delta"]
        elapsed = time.time() - iter_start

        # Decision: keep or rollback?
        improved = current_score > best_score
        if improved:
            best_score = current_score
            best_params = current_params.clone()
            best_iteration = iteration
            improve_count += 1
            status = "IMPROVED"
        else:
            current_params = best_params.clone()  # rollback
            rollback_count += 1
            status = "ROLLBACK"

        # Log
        cats = scores["categories"]
        cat_str = " | ".join(f"{c[:4]}:{cats[c]['delta']:+.2f}" for c in sorted(cats))
        history.append({
            "iteration": iteration,
            "eaam": current_score,
            "rag": scores["overall_rag"],
            "delta": delta_vs_rag,
            "status": status,
            "categories": {c: cats[c]["delta"] for c in cats},
        })

        bar = "+" * int(max(0, delta_vs_rag) * 40) + "-" * int(max(0, -delta_vs_rag) * 40)
        print(f"  [{iteration:3d}/{n_iterations}] EAAM={current_score:.4f} RAG={scores['overall_rag']:.4f} "
              f"Δ={delta_vs_rag:+.4f} [{bar[:20]:20}] {status:8} ({elapsed:.1f}s) | {cat_str}")

        # Mutate for next iteration
        current_params = mutate_params(current_params, scores, iteration)

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"  Iterations:    {n_iterations}")
    print(f"  Improvements:  {improve_count}")
    print(f"  Rollbacks:     {rollback_count}")
    print(f"  Best iteration: {best_iteration}")
    print(f"  Best EAAM:     {best_score:.4f}")
    print(f"  RAG baseline:  {history[0]['rag']:.4f}")
    print(f"  Best delta:    {best_score - history[0]['rag']:+.4f}")
    print(f"  Improvement:   {((best_score - history[0]['rag']) / max(history[0]['rag'], 0.001)) * 100:.1f}%")

    print(f"\n  Best Parameters:")
    for k, v in best_params.to_dict().items():
        print(f"    {k}: {v}")

    print(f"\n  Category deltas at best:")
    # Re-run best params to get category breakdown
    config = EAAMConfig()
    config.emotion.use_transformer = use_transformer
    config.graph.persist_path = tempfile.mkdtemp()
    config.vector.persist_path = tempfile.mkdtemp()
    store = MemoryStore(config)
    pipeline = EncodingPipeline(store, shared_emotion_encoder, config)
    pipeline._build_semantic_edges = make_patched_semantic(best_params, pipeline)
    pipeline._build_emotional_edges = make_patched_emotional(best_params, pipeline)
    retriever = AssociativeRetriever(store, shared_emotion_encoder, config.retrieval)
    rag = BaselineRAGRetriever(store)
    tag_index = {}
    for fixture in MEMORY_CORPUS:
        mem = pipeline.encode(content=fixture.content, conversation_id=fixture.conversation_id,
                              topic=fixture.topic, role=fixture.role)
        tag_index[mem.id] = set(fixture.tags)
    final_scores = score_benchmark(store, shared_emotion_encoder, pipeline, retriever, rag, best_params, tag_index)
    for cat, data in sorted(final_scores["categories"].items()):
        winner = "EAAM" if data["delta"] > 0.01 else ("RAG" if data["delta"] < -0.01 else "TIE")
        print(f"    {cat:25} RAG={data['rag']:.3f}  EAAM={data['eaam']:.3f}  Δ={data['delta']:+.3f}  {winner}")

    print(f"\n  Progress trace (every 10th iteration):")
    for h in history:
        if h["iteration"] % 10 == 0 or h["iteration"] == 1:
            print(f"    [{h['iteration']:3d}] EAAM={h['eaam']:.4f} Δ={h['delta']:+.4f} {h['status']}")

    # Save best params
    import yaml
    params_path = str(Path(__file__).resolve().parent.parent / "config" / "best_params.yaml")
    with open(params_path, "w") as f:
        yaml.dump(best_params.to_dict(), f, default_flow_style=False)
    print(f"\n  Best params saved to: {params_path}")


if __name__ == "__main__":
    n = 100
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n = int(sys.argv[1])
    use_t = "--transformer" in sys.argv or "-t" in sys.argv
    if not use_t:
        use_t = True  # default to transformer
    run_optimization(n_iterations=n, use_transformer=use_t)
