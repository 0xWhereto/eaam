"""Run the XO-ANON-XO loop on EAAM. Usage: python3 tests/run_xo_loop.py [iterations]"""

from __future__ import annotations
import math, random, sys, tempfile
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.xo_loop import RetrieverConfig, run_xo_loop
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever, PathwayResult
from eaam.models import VAD, Edge, EdgeType, Memory, RetrievalResult
from eaam.store.memory_store import MemoryStore

from tests.benchmark import (
    MEMORY_CORPUS, QUERY_SCENARIOS, BaselineRAGRetriever,
    precision_at_k, recall_at_k, ndcg_at_k, anti_tag_score,
)

print("Loading transformer model...")
_cfg = EAAMConfig(); _cfg.emotion.use_transformer = True
SHARED_EMOTION = EmotionEncoder(_cfg.emotion)
print("Model loaded.\n")


# ============================================================================
# VAD SIMILARITY IMPLEMENTATIONS
# ============================================================================

def vad_similarity(a: VAD, b: VAD, method: str = "euclidean") -> float:
    if method == "cosine":
        # Shift valence to [0,2] so all dims are non-negative for meaningful cosine
        av = [a.valence + 1, a.arousal, a.dominance]
        bv = [b.valence + 1, b.arousal, b.dominance]
        dot = sum(x * y for x, y in zip(av, bv))
        mag_a = math.sqrt(sum(x ** 2 for x in av)) or 1e-9
        mag_b = math.sqrt(sum(x ** 2 for x in bv)) or 1e-9
        return max(0, dot / (mag_a * mag_b))
    elif method == "manhattan":
        dist = abs(a.valence - b.valence) + abs(a.arousal - b.arousal) + abs(a.dominance - b.dominance)
        max_dist = 2 + 1 + 1  # max manhattan in VAD space
        return 1.0 - (dist / max_dist)
    else:  # euclidean (default)
        return a.similarity(b)


# ============================================================================
# FULL RETRIEVAL WITH STRUCTURAL OPTIONS
# ============================================================================

def retrieve_with_config(
    store: MemoryStore, retriever: AssociativeRetriever,
    rc: RetrieverConfig, query: str, k: int, emotional_context: str | None,
) -> list[RetrievalResult]:
    """Run retrieval with all structural + numeric options from RetrieverConfig."""

    # Determine current emotion
    if emotional_context:
        current_emotion = SHARED_EMOTION.encode(emotional_context)
    else:
        current_emotion = SHARED_EMOTION.encode(query)

    sim_fn = rc.vad_similarity_fn

    # --- PATHWAY 1: Hippocampal ---
    hippocampal = []
    hits = store.semantic_search(query, n=20)
    for mem_id, sim in hits:
        mem = store.get(mem_id)
        if mem and sim > 0.1:
            hippocampal.append(PathwayResult(memory=mem, score=sim, pathway="hippocampal"))

    # --- PATHWAY 2: Amygdalar ---
    amygdalar = []
    all_mems = store.graph.get_all_memories()
    for mem in all_mems:
        emo_sim = vad_similarity(current_emotion, mem.emotion, sim_fn)
        if emo_sim >= rc.amygdalar_emo_threshold:
            act = mem.effective_activation()
            score = emo_sim * rc.amygdalar_emo_weight + act * rc.amygdalar_act_weight
            amygdalar.append(PathwayResult(memory=mem, score=score, pathway="amygdalar"))
    amygdalar.sort(key=lambda r: -r.score)

    # Amygdalar diversity: deduplicate by VAD region
    if rc.amygdalar_diversity and len(amygdalar) > k:
        diverse = []
        seen_regions = set()
        for pr in amygdalar:
            region = (round(pr.memory.emotion.valence, 1), round(pr.memory.emotion.arousal, 1))
            if region not in seen_regions:
                diverse.append(pr)
                seen_regions.add(region)
            if len(diverse) >= k * 2:
                break
        amygdalar = diverse

    amygdalar = amygdalar[:k]

    # --- PATHWAY 3: Spreading ---
    spreading = []
    seeds = store.semantic_search(query, n=rc.fan_out_limit * 3)
    activated: dict[str, tuple[float, list[str]]] = {}
    for mem_id, sim in (seeds or []):
        activated[mem_id] = (sim, [mem_id])

    edge_type_multiplier = {"semantic": 1.0, "emotional": 1.0, "temporal": 1.0}
    if rc.spreading_edge_weighting == "prefer_emotional":
        edge_type_multiplier["emotional"] = 1.5
        edge_type_multiplier["semantic"] = 0.8
    elif rc.spreading_edge_weighting == "prefer_semantic":
        edge_type_multiplier["semantic"] = 1.5
        edge_type_multiplier["emotional"] = 0.7

    for hop in range(2):
        decay = rc.hop_decay ** (hop + 1)
        new_act: dict[str, tuple[float, list[str]]] = {}
        for node_id, (act, path) in list(activated.items()):
            if act < 0.03: continue
            neighbors = store.get_neighbors(node_id)
            neighbors.sort(key=lambda x: -x[1].weight)
            for neighbor, edge in neighbors[:rc.fan_out_limit]:
                if neighbor.id in set(path): continue
                etype = edge.edge_type.value if hasattr(edge.edge_type, 'value') else str(edge.edge_type)
                emult = edge_type_multiplier.get(etype, 1.0)
                spread = act * edge.weight * emult * decay
                emo_sim = vad_similarity(current_emotion, neighbor.emotion, sim_fn)
                spread *= (1.0 + rc.emotion_boost_spreading * emo_sim)
                new_path = path + [neighbor.id]
                ea, ep = new_act.get(neighbor.id, (0.0, []))
                combined = min(ea + spread, rc.spreading_cap)
                new_act[neighbor.id] = (combined, new_path if spread > ea else ep)
        for nid, (a, p) in new_act.items():
            oa, op = activated.get(nid, (0.0, []))
            activated[nid] = (min(oa + a, rc.spreading_cap), p if a > oa else op)

    for mem_id, (act, path) in activated.items():
        if len(path) <= 1: continue
        mem = store.get(mem_id)
        if mem and act > 0.05:
            spreading.append(PathwayResult(memory=mem, score=act, pathway="spreading", path=path))
    spreading.sort(key=lambda r: -r.score)
    spreading = spreading[:k]

    # --- PATHWAY 4: Involuntary ---
    involuntary = []
    if rc.involuntary_mode == "arousal_hub":
        hubs = sorted(all_mems, key=lambda m: m.emotion.arousal * m.effective_activation(), reverse=True)
    elif rc.involuntary_mode == "activation_hub":
        hubs = sorted(all_mems, key=lambda m: m.effective_activation(), reverse=True)
    else:  # emotional_outlier
        neutral = VAD.neutral()
        hubs = sorted(all_mems, key=lambda m: -abs(m.emotion.valence) - m.emotion.arousal)

    for mem in hubs[:k * 3]:
        if mem.is_reflection: continue
        emo_match = vad_similarity(current_emotion, mem.emotion, sim_fn)
        qw = set(query.lower().split()); cw = set(mem.content.lower().split())
        overlap = len(qw & cw) / max(len(qw), 1)
        cue = max(emo_match * 0.6, overlap * 0.4)
        fire = (mem.effective_activation() * rc.involuntary_activation_weight +
                mem.emotion.arousal * rc.involuntary_arousal_weight +
                cue * rc.involuntary_cue_weight)
        if fire > rc.involuntary_fire_threshold:
            involuntary.append(PathwayResult(memory=mem, score=fire, pathway="involuntary"))
    involuntary.sort(key=lambda r: -r.score)
    involuntary = involuntary[:max(2, k // 2)]

    # --- QUERY CLASSIFICATION ---
    query_type = "mixed"
    if rc.query_classifier == "keyword":
        emo_words = {"happy","sad","angry","scared","frustrated","excited","worried","anxious",
                     "terrified","furious","depressed","thrilled","overwhelmed","panicking","devastated"}
        if any(w in query.lower() for w in emo_words):
            query_type = "emotional"
        else:
            query_type = "semantic"
    elif rc.query_classifier == "emotion_strength":
        if current_emotion.arousal > 0.6:
            query_type = "emotional"
        elif current_emotion.arousal < 0.3:
            query_type = "semantic"

    # --- MERGE ---
    seen: set[str] = set()
    merged: list[RetrievalResult] = []

    def add(candidates, n):
        added = 0
        for pr in candidates:
            if pr.memory.id in seen or added >= n: continue
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

    if rc.merge_strategy == "tournament":
        # Pure tournament — all pathways compete by score
        all_results = hippocampal + amygdalar + spreading + involuntary
        all_results.sort(key=lambda r: -r.score)
        add(all_results, k)
    elif rc.merge_strategy == "adaptive":
        # Adaptive based on query type
        if query_type == "semantic":
            hip_n = max(2, int(k * 0.5))
            add(hippocampal, hip_n)
            if amygdalar: add(amygdalar, 1)
            if spreading: add(spreading, 1)
        elif query_type == "emotional":
            add(hippocampal, 1)
            if amygdalar: add(amygdalar, max(2, int(k * 0.4)))
            if spreading: add(spreading, 1)
            if involuntary: add(involuntary, 1)
        else:
            hip_n = max(1, int(k * rc.hippocampal_ratio))
            add(hippocampal, hip_n)
            if amygdalar: add(amygdalar, 1)
            if spreading: add(spreading, 1)
            if involuntary: add(involuntary, 1)
        # Fill remaining
        rem = k - len(merged)
        if rem > 0:
            leftovers = [pr for lst in [hippocampal, amygdalar, spreading, involuntary]
                         for pr in lst if pr.memory.id not in seen]
            leftovers.sort(key=lambda r: -r.score)
            add(leftovers, rem)
    else:
        # Fixed slots (default)
        hip_n = max(1, int(k * rc.hippocampal_ratio))
        if rc.hippocampal_boost_if_semantic and query_type == "semantic":
            hip_n = min(k - 1, hip_n + 2)
        add(hippocampal, hip_n)
        if amygdalar: add(amygdalar, 1)
        if spreading: add(spreading, 1)
        if involuntary: add(involuntary, 1)
        rem = k - len(merged)
        if rem > 0:
            leftovers = [pr for lst in [hippocampal, amygdalar, spreading, involuntary]
                         for pr in lst if pr.memory.id not in seen]
            leftovers.sort(key=lambda r: -r.score)
            add(leftovers, rem)

    # Reconsolidation
    if rc.reconsolidation_enabled:
        for r in merged[:k]:
            r.memory.emotion.valence += rc.reconsolidation_rate * (current_emotion.valence - r.memory.emotion.valence)
            r.memory.emotion.arousal += rc.reconsolidation_rate * (current_emotion.arousal - r.memory.emotion.arousal)
            r.memory.emotion.__post_init__()

    merged.sort(key=lambda r: -r.score)
    return merged[:k]


# ============================================================================
# SCORING FUNCTION
# ============================================================================

def score_config(rc: RetrieverConfig) -> dict:
    """Build fresh store, populate, score all scenarios."""
    config = EAAMConfig()
    config.emotion.use_transformer = True
    config.graph.persist_path = tempfile.mkdtemp()
    config.vector.persist_path = tempfile.mkdtemp()

    store = MemoryStore(config)
    pipeline = EncodingPipeline(store, SHARED_EMOTION, config)
    retriever = AssociativeRetriever(store, SHARED_EMOTION, config.retrieval)
    rag = BaselineRAGRetriever(store)

    # Patch encoding thresholds
    def make_sem(p, pip):
        def f(memory):
            for mid, sim in pip.store.semantic_search(memory.content, n=10):
                if mid != memory.id and sim >= p.semantic_edge_threshold:
                    pip.store.add_edge(Edge(source_id=memory.id, target_id=mid,
                                            edge_type=EdgeType.SEMANTIC, weight=sim))
        return f
    def make_emo(p, pip):
        def f(memory):
            neighbors = pip.store.emotional_search(memory.emotion, threshold=p.emotional_edge_threshold, limit=8)
            sem_cands = dict(pip.store.semantic_search(memory.content, n=20))
            for nb, es in neighbors:
                if nb.id != memory.id and sem_cands.get(nb.id, 0) < p.emotional_semantic_filter:
                    pip.store.add_edge(Edge(source_id=memory.id, target_id=nb.id,
                                            edge_type=EdgeType.EMOTIONAL, weight=es))
        return f
    pipeline._build_semantic_edges = make_sem(rc, pipeline)
    pipeline._build_emotional_edges = make_emo(rc, pipeline)

    tag_index = {}
    for fix in MEMORY_CORPUS:
        mem = pipeline.encode(content=fix.content, conversation_id=fix.conversation_id,
                              topic=fix.topic, role=fix.role)
        tag_index[mem.id] = set(fix.tags)

    K = 5
    cat_rag: dict[str, list[float]] = {}
    cat_eaam: dict[str, list[float]] = {}

    for sc in QUERY_SCENARIOS:
        # RAG
        rr = rag.retrieve(sc.query, k=K)
        rt = [tag_index.get(r.memory.id, set()) for r in rr]
        # EAAM
        er = retrieve_with_config(store, retriever, rc, sc.query, K, sc.emotional_context)
        et = [tag_index.get(r.memory.id, set()) for r in er]

        for tl, cd in [(rt, cat_rag), (et, cat_eaam)]:
            p = precision_at_k(tl, sc.expected_tags, K)
            r = recall_at_k(tl, sc.expected_tags, K)
            n = ndcg_at_k(tl, sc.expected_tags, K)
            a = anti_tag_score(tl, sc.anti_tags, K)
            cd.setdefault(sc.category, []).append((p + r + n) / 3 - a * 0.5)

    result = {"categories": {}}
    tr, te, tn = 0.0, 0.0, 0
    for cat in sorted(set(list(cat_rag.keys()) + list(cat_eaam.keys()))):
        ar = sum(cat_rag.get(cat, [0])) / max(len(cat_rag.get(cat, [1])), 1)
        ae = sum(cat_eaam.get(cat, [0])) / max(len(cat_eaam.get(cat, [1])), 1)
        result["categories"][cat] = {"rag": ar, "eaam": ae, "delta": ae - ar}
        tr += sum(cat_rag.get(cat, [])); te += sum(cat_eaam.get(cat, [])); tn += len(cat_rag.get(cat, []))

    result["overall_rag"] = tr / max(tn, 1)
    result["overall_eaam"] = te / max(tn, 1)
    result["delta"] = result["overall_eaam"] - result["overall_rag"]
    return result


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    best_config, best_score, ledger = run_xo_loop(score_config, n_iterations=n)

    import yaml
    out = {k: v for k, v in vars(best_config).items()}
    with open(_ROOT / "config" / "xo_best_config.yaml", "w") as f:
        yaml.dump(out, f, default_flow_style=False)
    print(f"\nBest config saved to config/xo_best_config.yaml")
