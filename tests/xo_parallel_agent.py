"""XO Parallel Agent — one lane of the 10-agent parallel optimization.

Takes a --seed argument that determines:
1. Starting config perturbation (different region of param space)
2. Strategy exploration order (shuffled by seed)
3. Random compound generation seed

Outputs results to a JSON file for the orchestrator to merge.

Usage: python3 tests/xo_parallel_agent.py --seed N --iterations 100 --output /path/to/result.json
"""
from __future__ import annotations

import hashlib, json, math, random, sys, tempfile, time, os
from dataclasses import dataclass, field
from typing import Any, Callable

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever, PathwayResult
from eaam.models import VAD, Edge, EdgeType, Memory, RetrievalResult
from eaam.store.memory_store import MemoryStore

from tests.benchmark import (
    MEMORY_CORPUS, QUERY_SCENARIOS, BaselineRAGRetriever,
    precision_at_k, recall_at_k, ndcg_at_k, anti_tag_score,
)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class RC:
    """RetrieverConfig — all tunable state."""
    hop_decay: float = 0.50
    emotion_boost_spreading: float = 0.30
    spreading_cap: float = 0.8274
    fan_out_limit: int = 5
    hippocampal_ratio: float = 0.33
    amygdalar_emo_threshold: float = 0.70
    amygdalar_emo_weight: float = 0.6774
    amygdalar_act_weight: float = 0.30
    involuntary_fire_threshold: float = 0.35
    involuntary_arousal_weight: float = 0.233
    involuntary_activation_weight: float = 0.40
    involuntary_cue_weight: float = 0.30
    semantic_edge_threshold: float = 0.30
    emotional_edge_threshold: float = 0.75
    emotional_semantic_filter: float = 0.50
    # Structural
    spreading_edge_weighting: str = "prefer_semantic"
    involuntary_mode: str = "emotional_outlier"
    merge_strategy: str = "fixed_slots"
    query_classifier: str = "none"
    hippocampal_boost_if_semantic: bool = False
    amygdalar_diversity: bool = False
    reconsolidation_enabled: bool = True
    reconsolidation_rate: float = 0.05
    # v2 additions for expanded exploration
    spreading_hops: int = 2
    amygdalar_limit_multiplier: float = 2.0
    involuntary_pool_multiplier: float = 3.0
    semantic_search_pool: int = 20

    def fingerprint(self) -> str:
        d = {}
        for k, v in sorted(vars(self).items()):
            d[k] = round(v, 4) if isinstance(v, float) else v
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def clone(self):
        return RC(**vars(self))

    def to_dict(self):
        return dict(vars(self))


# ============================================================================
# EXPANDED STRATEGY GENERATOR
# ============================================================================

def generate_strategies(seed: int) -> list[dict]:
    """Generate a large, seed-specific strategy library.

    Returns list of {id, name, category, mutations: {param: value}}
    """
    rng = random.Random(seed)
    strategies = []

    # --- Fine-grained numeric sweeps ---
    numeric_params = {
        "hop_decay": (0.10, 0.70, 0.025),
        "emotion_boost_spreading": (0.05, 0.70, 0.05),
        "spreading_cap": (0.30, 0.95, 0.05),
        "hippocampal_ratio": (0.15, 0.60, 0.03),
        "amygdalar_emo_threshold": (0.45, 0.85, 0.03),
        "amygdalar_emo_weight": (0.40, 0.90, 0.03),
        "amygdalar_act_weight": (0.10, 0.55, 0.05),
        "involuntary_fire_threshold": (0.15, 0.55, 0.03),
        "involuntary_arousal_weight": (0.10, 0.50, 0.05),
        "involuntary_activation_weight": (0.15, 0.60, 0.05),
        "involuntary_cue_weight": (0.10, 0.50, 0.05),
        "semantic_edge_threshold": (0.15, 0.50, 0.05),
        "emotional_edge_threshold": (0.50, 0.90, 0.05),
        "emotional_semantic_filter": (0.25, 0.75, 0.05),
        "reconsolidation_rate": (0.01, 0.15, 0.02),
        "amygdalar_limit_multiplier": (1.0, 4.0, 0.5),
        "involuntary_pool_multiplier": (1.5, 5.0, 0.5),
    }
    int_params = {
        "fan_out_limit": (2, 8),
        "spreading_hops": (1, 4),
        "semantic_search_pool": (10, 40),
    }
    for name, (lo, hi, step) in numeric_params.items():
        v = lo
        while v <= hi + 0.001:
            val = round(v, 4)
            strategies.append({
                "id": f"p_{name}_{val}", "name": f"{name}={val}",
                "category": "numeric", "mutations": {name: val},
            })
            v += step
    for name, (lo, hi) in int_params.items():
        for val in range(lo, hi + 1):
            strategies.append({
                "id": f"p_{name}_{val}", "name": f"{name}={val}",
                "category": "numeric", "mutations": {name: val},
            })

    # --- Structural strategies ---
    for opt, vals in [
        ("spreading_edge_weighting", ["uniform", "prefer_emotional", "prefer_semantic"]),
        ("involuntary_mode", ["arousal_hub", "activation_hub", "emotional_outlier"]),
        ("merge_strategy", ["fixed_slots", "adaptive", "tournament"]),
        ("query_classifier", ["none", "keyword", "emotion_strength"]),
    ]:
        for val in vals:
            strategies.append({
                "id": f"s_{opt}_{val}", "name": f"{opt}={val}",
                "category": "structural", "mutations": {opt: val},
            })

    for flag in ["hippocampal_boost_if_semantic", "amygdalar_diversity", "reconsolidation_enabled"]:
        for val in [True, False]:
            strategies.append({
                "id": f"f_{flag}_{val}", "name": f"{flag}={val}",
                "category": "structural", "mutations": {flag: val},
            })

    # --- Random compound strategies (seed-specific) ---
    all_param_names = list(numeric_params.keys()) + list(int_params.keys())
    for i in range(80):
        n_params = rng.randint(2, 4)
        chosen = rng.sample(all_param_names, min(n_params, len(all_param_names)))
        mutations = {}
        for p in chosen:
            if p in numeric_params:
                lo, hi, _ = numeric_params[p]
                mutations[p] = round(rng.uniform(lo, hi), 4)
            else:
                lo, hi = int_params[p]
                mutations[p] = rng.randint(lo, hi)
        sid = hashlib.sha1(json.dumps(mutations, sort_keys=True).encode()).hexdigest()[:8]
        strategies.append({
            "id": f"c_rand_{sid}", "name": f"compound({', '.join(chosen)})",
            "category": "compound", "mutations": mutations,
        })

    # Shuffle by seed for diverse exploration order
    rng.shuffle(strategies)
    return strategies


# ============================================================================
# SCORING (same as run_xo_loop but self-contained)
# ============================================================================

def vad_sim(a, b, method="euclidean"):
    if method == "cosine":
        av = [a.valence+1, a.arousal, a.dominance]
        bv = [b.valence+1, b.arousal, b.dominance]
        dot = sum(x*y for x,y in zip(av,bv))
        return max(0, dot/((sum(x**2 for x in av)**0.5 or 1e-9)*(sum(x**2 for x in bv)**0.5 or 1e-9)))
    elif method == "manhattan":
        return 1.0 - (abs(a.valence-b.valence)+abs(a.arousal-b.arousal)+abs(a.dominance-b.dominance))/4
    return a.similarity(b)


def score_rc(rc: RC, emotion_encoder: EmotionEncoder) -> dict:
    cfg = EAAMConfig()
    cfg.emotion.use_transformer = True
    cfg.graph.persist_path = tempfile.mkdtemp()
    cfg.vector.persist_path = tempfile.mkdtemp()

    store = MemoryStore(cfg)
    pipe = EncodingPipeline(store, emotion_encoder, cfg)
    retriever = AssociativeRetriever(store, emotion_encoder, cfg.retrieval)
    rag = BaselineRAGRetriever(store)

    def psem(memory):
        for mid, sim in pipe.store.semantic_search(memory.content, n=10):
            if mid != memory.id and sim >= rc.semantic_edge_threshold:
                pipe.store.add_edge(Edge(source_id=memory.id, target_id=mid, edge_type=EdgeType.SEMANTIC, weight=sim))
    def pemo(memory):
        nbs = pipe.store.emotional_search(memory.emotion, threshold=rc.emotional_edge_threshold, limit=8)
        sc = dict(pipe.store.semantic_search(memory.content, n=20))
        for nb, es in nbs:
            if nb.id != memory.id and sc.get(nb.id, 0) < rc.emotional_semantic_filter:
                pipe.store.add_edge(Edge(source_id=memory.id, target_id=nb.id, edge_type=EdgeType.EMOTIONAL, weight=es))
    pipe._build_semantic_edges = psem
    pipe._build_emotional_edges = pemo

    tags = {}
    for fix in MEMORY_CORPUS:
        m = pipe.encode(content=fix.content, conversation_id=fix.conversation_id, topic=fix.topic, role=fix.role)
        tags[m.id] = set(fix.tags)

    K = 5
    cr, ce = {}, {}
    for sc in QUERY_SCENARIOS:
        rr = rag.retrieve(sc.query, k=K)
        rt = [tags.get(r.memory.id, set()) for r in rr]
        er = _retrieve(store, rc, sc.query, K, sc.emotional_context, emotion_encoder)
        et = [tags.get(r.memory.id, set()) for r in er]
        for tl, cd in [(rt, cr), (et, ce)]:
            p = precision_at_k(tl, sc.expected_tags, K)
            r = recall_at_k(tl, sc.expected_tags, K)
            n = ndcg_at_k(tl, sc.expected_tags, K)
            a = anti_tag_score(tl, sc.anti_tags, K)
            cd.setdefault(sc.category, []).append((p+r+n)/3 - a*0.5)

    result = {"categories": {}}
    tr, te, tn = 0.0, 0.0, 0
    for cat in sorted(set(list(cr.keys())+list(ce.keys()))):
        ar = sum(cr.get(cat,[0]))/max(len(cr.get(cat,[1])),1)
        ae = sum(ce.get(cat,[0]))/max(len(ce.get(cat,[1])),1)
        result["categories"][cat] = {"rag": ar, "eaam": ae, "delta": ae-ar}
        tr += sum(cr.get(cat,[])); te += sum(ce.get(cat,[])); tn += len(cr.get(cat,[]))
    result["overall_rag"] = tr/max(tn,1)
    result["overall_eaam"] = te/max(tn,1)
    result["delta"] = result["overall_eaam"]-result["overall_rag"]
    return result


def _retrieve(store, rc, query, k, emotional_context, ee):
    cur_emo = ee.encode(emotional_context) if emotional_context else ee.encode(query)
    sf = rc.spreading_edge_weighting
    sim_fn = "euclidean"

    # Hippocampal
    hip = []
    for mid, sim in store.semantic_search(query, n=rc.semantic_search_pool):
        m = store.get(mid)
        if m and sim > 0.1: hip.append(PathwayResult(memory=m, score=sim, pathway="hippocampal"))

    # Amygdalar
    amy = []
    for m in store.graph.get_all_memories():
        es = vad_sim(cur_emo, m.emotion, sim_fn)
        if es >= rc.amygdalar_emo_threshold:
            sc = es*rc.amygdalar_emo_weight + m.effective_activation()*rc.amygdalar_act_weight
            amy.append(PathwayResult(memory=m, score=sc, pathway="amygdalar"))
    amy.sort(key=lambda r: -r.score)
    if rc.amygdalar_diversity:
        diverse, seen_r = [], set()
        for pr in amy:
            rg = (round(pr.memory.emotion.valence, 1), round(pr.memory.emotion.arousal, 1))
            if rg not in seen_r: diverse.append(pr); seen_r.add(rg)
        amy = diverse
    amy = amy[:int(k*rc.amygdalar_limit_multiplier)]

    # Spreading
    spr = []
    seeds = store.semantic_search(query, n=rc.fan_out_limit*3)
    act = {mid: (sim, [mid]) for mid, sim in (seeds or [])}
    emult = {"semantic": 1.0, "emotional": 1.0, "temporal": 1.0}
    if sf == "prefer_emotional": emult["emotional"]=1.5; emult["semantic"]=0.8
    elif sf == "prefer_semantic": emult["semantic"]=1.5; emult["emotional"]=0.7
    for hop in range(rc.spreading_hops):
        decay = rc.hop_decay**(hop+1)
        new = {}
        for nid, (a, path) in list(act.items()):
            if a < 0.03: continue
            nbs = store.get_neighbors(nid)
            nbs.sort(key=lambda x: -x[1].weight)
            for nb, edge in nbs[:rc.fan_out_limit]:
                if nb.id in set(path): continue
                et = edge.edge_type.value if hasattr(edge.edge_type,'value') else str(edge.edge_type)
                sp = a*edge.weight*emult.get(et,1.0)*decay
                sp *= (1.0+rc.emotion_boost_spreading*vad_sim(cur_emo,nb.emotion,sim_fn))
                np_ = path+[nb.id]
                ea, ep = new.get(nb.id, (0.0, []))
                new[nb.id] = (min(ea+sp, rc.spreading_cap), np_ if sp > ea else ep)
        for nid, (a, p) in new.items():
            oa, op = act.get(nid, (0.0, []))
            act[nid] = (min(oa+a, rc.spreading_cap), p if a > oa else op)
    for mid, (a, path) in act.items():
        if len(path) <= 1: continue
        m = store.get(mid)
        if m and a > 0.05: spr.append(PathwayResult(memory=m, score=a, pathway="spreading", path=path))
    spr.sort(key=lambda r: -r.score); spr = spr[:k]

    # Involuntary
    inv = []
    mems = store.graph.get_all_memories()
    if rc.involuntary_mode == "activation_hub":
        hubs = sorted(mems, key=lambda m: m.effective_activation(), reverse=True)
    elif rc.involuntary_mode == "emotional_outlier":
        hubs = sorted(mems, key=lambda m: -abs(m.emotion.valence)-m.emotion.arousal)
    else:
        hubs = sorted(mems, key=lambda m: m.emotion.arousal*m.effective_activation(), reverse=True)
    for m in hubs[:int(k*rc.involuntary_pool_multiplier)]:
        if m.is_reflection: continue
        em = vad_sim(cur_emo, m.emotion, sim_fn)
        qw = set(query.lower().split()); cw = set(m.content.lower().split())
        ov = len(qw&cw)/max(len(qw),1)
        cue = max(em*0.6, ov*0.4)
        fire = m.effective_activation()*rc.involuntary_activation_weight + m.emotion.arousal*rc.involuntary_arousal_weight + cue*rc.involuntary_cue_weight
        if fire > rc.involuntary_fire_threshold:
            inv.append(PathwayResult(memory=m, score=fire, pathway="involuntary"))
    inv.sort(key=lambda r: -r.score); inv = inv[:max(2, k//2)]

    # Query classification
    qt = "mixed"
    if rc.query_classifier == "keyword":
        ew = {"happy","sad","angry","scared","frustrated","excited","worried","anxious","terrified","furious","depressed","overwhelmed","panicking","devastated"}
        qt = "emotional" if any(w in query.lower() for w in ew) else "semantic"
    elif rc.query_classifier == "emotion_strength":
        qt = "emotional" if cur_emo.arousal > 0.6 else ("semantic" if cur_emo.arousal < 0.3 else "mixed")

    # Merge
    seen = set(); merged = []
    def add(cands, n):
        added = 0
        for pr in cands:
            if pr.memory.id in seen or added >= n: continue
            seen.add(pr.memory.id)
            merged.append(RetrievalResult(memory=pr.memory, score=pr.score,
                semantic_score=pr.score if pr.pathway=="hippocampal" else 0.0,
                emotional_score=pr.score if pr.pathway=="amygdalar" else 0.0,
                activation_score=pr.score if pr.pathway=="involuntary" else 0.0,
                spreading_score=pr.score if pr.pathway=="spreading" else 0.0,
                path=pr.path or [pr.memory.id]))
            added += 1

    if rc.merge_strategy == "tournament":
        all_r = hip+amy+spr+inv; all_r.sort(key=lambda r: -r.score); add(all_r, k)
    elif rc.merge_strategy == "adaptive":
        if qt == "semantic": add(hip, max(2,int(k*0.5))); add(amy[:1],1); add(spr[:1],1)
        elif qt == "emotional": add(hip[:1],1); add(amy, max(2,int(k*0.4))); add(spr[:1],1); add(inv[:1],1)
        else: add(hip, max(1,int(k*rc.hippocampal_ratio))); add(amy[:1],1); add(spr[:1],1); add(inv[:1],1)
        left = [pr for l in [hip,amy,spr,inv] for pr in l if pr.memory.id not in seen]
        left.sort(key=lambda r: -r.score); add(left, k-len(merged))
    else:
        hn = max(1,int(k*rc.hippocampal_ratio))
        if rc.hippocampal_boost_if_semantic and qt=="semantic": hn=min(k-1,hn+2)
        add(hip,hn); add(amy[:1],1); add(spr[:1],1); add(inv[:1],1)
        left = [pr for l in [hip,amy,spr,inv] for pr in l if pr.memory.id not in seen]
        left.sort(key=lambda r: -r.score); add(left, k-len(merged))

    if rc.reconsolidation_enabled:
        for r in merged[:k]:
            r.memory.emotion.valence += rc.reconsolidation_rate*(cur_emo.valence-r.memory.emotion.valence)
            r.memory.emotion.arousal += rc.reconsolidation_rate*(cur_emo.arousal-r.memory.emotion.arousal)
            r.memory.emotion.__post_init__()

    merged.sort(key=lambda r: -r.score)
    return merged[:k]


# ============================================================================
# MAIN AGENT LOOP
# ============================================================================

def run_agent(seed: int, n_iterations: int, output_path: str):
    rng = random.Random(seed)
    print(f"[Agent {seed}] Loading model...")
    cfg = EAAMConfig(); cfg.emotion.use_transformer = True
    ee = EmotionEncoder(cfg.emotion)
    print(f"[Agent {seed}] Generating strategies...")
    strategies = generate_strategies(seed)
    print(f"[Agent {seed}] {len(strategies)} strategies generated")

    # Start from best known config
    config = RC()
    best = config.clone()

    # Apply seed-based starting perturbation
    perturbations = [
        {"hop_decay": 0.40 + seed*0.02},
        {"amygdalar_emo_weight": 0.60 + seed*0.02},
        {"spreading_cap": 0.70 + seed*0.02},
        {"involuntary_fire_threshold": 0.25 + seed*0.02},
        {"emotion_boost_spreading": 0.20 + seed*0.03},
        {"emotional_edge_threshold": 0.60 + seed*0.03},
        {"hippocampal_ratio": 0.25 + seed*0.02},
        {"fan_out_limit": 3 + seed % 5},
        {"reconsolidation_rate": 0.03 + seed*0.01},
        {"amygdalar_emo_threshold": 0.55 + seed*0.03},
    ]
    if seed < len(perturbations):
        for k, v in perturbations[seed].items():
            if isinstance(getattr(config, k), int):
                setattr(config, k, int(v))
            else:
                setattr(config, k, round(v, 4))

    print(f"[Agent {seed}] Running baseline...")
    baseline = score_rc(config, ee)
    best_score = baseline["overall_eaam"]
    print(f"[Agent {seed}] Baseline: EAAM={best_score:.4f} Δ={baseline['delta']:+.4f}")

    fingerprints = set()
    committed = 0
    history = []
    strategy_idx = 0

    for it in range(1, n_iterations + 1):
        if strategy_idx >= len(strategies):
            # Wrap around with random shuffle
            rng.shuffle(strategies)
            strategy_idx = 0

        strat = strategies[strategy_idx]
        strategy_idx += 1

        # Apply mutations
        candidate = config.clone()
        for k, v in strat["mutations"].items():
            if hasattr(candidate, k):
                setattr(candidate, k, v)

        fp = candidate.fingerprint()
        if fp in fingerprints:
            continue
        fingerprints.add(fp)

        # Score
        scores = score_rc(candidate, ee)
        new_score = scores["overall_eaam"]

        # Gates: no category can regress > 0.08 from baseline
        regression = False
        for cat in scores.get("categories", {}):
            if cat in baseline.get("categories", {}):
                if scores["categories"][cat]["delta"] < baseline["categories"][cat]["delta"] - 0.08:
                    regression = True
                    break

        if regression or new_score < best_score - 0.005:
            status = "REJECT"
        elif new_score > best_score + 0.0005:
            config = candidate
            best = config.clone()
            best_score = new_score
            baseline = scores
            committed += 1
            status = "COMMIT"
            history.append({
                "iteration": it, "strategy": strat["id"], "category": strat["category"],
                "score": best_score, "delta": scores["delta"],
                "mutations": strat["mutations"],
            })
        else:
            status = "FLAT"

        if it % 10 == 0 or status == "COMMIT":
            print(f"[Agent {seed}] [{it:3d}/{n_iterations}] {strat['id'][:40]:40} "
                  f"EAAM={best_score:.4f} Δ={baseline['delta']:+.4f} {status}")

    # Save results
    result = {
        "seed": seed,
        "best_score": best_score,
        "best_delta": baseline["delta"],
        "committed": committed,
        "iterations_run": min(strategy_idx, n_iterations),
        "best_config": best.to_dict(),
        "history": history,
        "final_categories": {c: d for c, d in baseline.get("categories", {}).items()},
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"[Agent {seed}] DONE: EAAM={best_score:.4f} Δ={baseline['delta']:+.4f} "
          f"commits={committed} -> {output_path}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    run_agent(args.seed, args.iterations, args.output)
