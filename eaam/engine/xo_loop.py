"""EAAM XO Loop — xoanonxoloop-inspired autonomous improvement engine.

Unlike the parameter-only autotuner, this loop proposes STRUCTURAL changes:
- Swap similarity functions
- Change merge strategies
- Add query-type classification
- Modify pathway formulas
- Compound multi-surface mutations

Each iteration follows the xoanonxoloop staged pattern:
  PLANNER → analyzes weakness → picks a strategy
  EDITOR  → applies the strategy → produces a modified retriever
  GATES   → scope → behavioral → coherence → commit/rollback
  LEDGER  → fingerprint + outcome tracking
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from eaam.models import VAD, Memory, RetrievalResult


# ============================================================================
# STRATEGY DEFINITIONS — the "mutations" our loop can propose
# ============================================================================

@dataclass
class Strategy:
    """A named, parameterized change to the retrieval system."""
    id: str
    name: str
    category: str  # "algorithmic", "structural", "formula", "compound"
    description: str
    apply_fn: Callable  # (retriever_config) -> retriever_config
    affects: list[str]  # which benchmark categories this targets
    risk: str = "safe"  # "safe", "guarded"


@dataclass
class RetrieverConfig:
    """All tunable state for one retriever run — both params AND algorithmic choices."""
    # --- Numeric params ---
    hop_decay: float = 0.3323
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

    # --- Algorithmic choices (the structural part) ---
    vad_similarity_fn: str = "euclidean"          # "euclidean", "cosine", "manhattan"
    merge_strategy: str = "fixed_slots"            # "fixed_slots", "adaptive", "tournament"
    query_classifier: str = "none"                 # "none", "keyword", "emotion_strength"
    spreading_edge_weighting: str = "uniform"      # "uniform", "prefer_emotional", "prefer_semantic"
    involuntary_mode: str = "arousal_hub"          # "arousal_hub", "activation_hub", "emotional_outlier"
    hippocampal_boost_if_semantic: bool = False     # give extra slots if query is highly semantic
    amygdalar_diversity: bool = False               # force emotional results to span different VAD regions
    spreading_bidirectional: bool = True            # traverse edges both directions
    reconsolidation_enabled: bool = True            # update emotions on retrieval
    reconsolidation_rate: float = 0.05

    def fingerprint(self) -> str:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sorted(vars(self).items())}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def clone(self) -> RetrieverConfig:
        return RetrieverConfig(**vars(self))


# ============================================================================
# STRATEGY LIBRARY — all possible mutations
# ============================================================================

def _build_strategy_library() -> list[Strategy]:
    """Build the full library of strategies the planner can choose from."""
    strategies = []

    # --- ALGORITHMIC: Similarity functions ---
    for fn in ["cosine", "manhattan"]:
        strategies.append(Strategy(
            id=f"vad_sim_{fn}", name=f"Switch VAD similarity to {fn}",
            category="algorithmic",
            description=f"Change VAD similarity from euclidean to {fn}. "
                        f"Cosine ignores magnitude, manhattan is L1 distance.",
            apply_fn=lambda c, f=fn: _set(c, "vad_similarity_fn", f),
            affects=["emotional_congruence", "mood_congruent", "cross_domain_leap"],
        ))

    # --- STRUCTURAL: Merge strategies ---
    strategies.append(Strategy(
        id="merge_adaptive", name="Adaptive merge (query-aware slot allocation)",
        category="structural",
        description="Allocate pathway slots based on query analysis: "
                    "emotional queries get more amygdalar slots, semantic queries get more hippocampal.",
        apply_fn=lambda c: _set(c, "merge_strategy", "adaptive"),
        affects=["semantic_baseline", "emotional_congruence", "mood_congruent"],
    ))
    strategies.append(Strategy(
        id="merge_tournament", name="Tournament merge (pathways compete by score)",
        category="structural",
        description="No guaranteed slots. All pathways compete. Top-k by raw score wins. "
                    "This can let a single dominant pathway take all slots.",
        apply_fn=lambda c: _set(c, "merge_strategy", "tournament"),
        affects=["semantic_baseline", "emotional_congruence"],
    ))

    # --- STRUCTURAL: Query classification ---
    strategies.append(Strategy(
        id="query_classify_keyword", name="Enable keyword-based query classification",
        category="structural",
        description="Detect if query contains emotional keywords. If yes, boost amygdalar. "
                    "If no, boost hippocampal.",
        apply_fn=lambda c: _set(c, "query_classifier", "keyword"),
        affects=["semantic_baseline", "emotional_congruence"],
    ))
    strategies.append(Strategy(
        id="query_classify_emotion", name="Enable emotion-strength query classification",
        category="structural",
        description="Measure emotional arousal of the query. High-arousal queries "
                    "get more emotional pathway influence.",
        apply_fn=lambda c: _set(c, "query_classifier", "emotion_strength"),
        affects=["emotional_congruence", "mood_congruent", "activation"],
    ))

    # --- FORMULA: Edge weighting in spreading ---
    strategies.append(Strategy(
        id="spread_prefer_emotional", name="Spreading prefers emotional edges",
        category="formula",
        description="During spreading, emotional edges get 1.5x weight multiplier. "
                    "This amplifies cross-domain leaps via shared emotion.",
        apply_fn=lambda c: _set(c, "spreading_edge_weighting", "prefer_emotional"),
        affects=["cross_domain_leap", "mood_congruent"],
    ))
    strategies.append(Strategy(
        id="spread_prefer_semantic", name="Spreading prefers semantic edges",
        category="formula",
        description="During spreading, semantic edges get 1.5x weight. "
                    "This keeps spreading closer to the original topic.",
        apply_fn=lambda c: _set(c, "spreading_edge_weighting", "prefer_semantic"),
        affects=["semantic_baseline", "cross_domain_leap"],
    ))

    # --- FORMULA: Involuntary modes ---
    strategies.append(Strategy(
        id="involuntary_activation_hub", name="Involuntary: activation hubs instead of arousal",
        category="formula",
        description="Involuntary memories fire based on access frequency and recency "
                    "rather than emotional arousal.",
        apply_fn=lambda c: _set(c, "involuntary_mode", "activation_hub"),
        affects=["activation"],
    ))
    strategies.append(Strategy(
        id="involuntary_emotional_outlier", name="Involuntary: emotional outliers",
        category="formula",
        description="Surface memories with the most EXTREME emotional signatures "
                    "(furthest from neutral) regardless of activation.",
        apply_fn=lambda c: _set(c, "involuntary_mode", "emotional_outlier"),
        affects=["activation", "cross_domain_leap"],
    ))

    # --- STRUCTURAL: Feature flags ---
    strategies.append(Strategy(
        id="hippocampal_boost", name="Boost hippocampal for semantic queries",
        category="structural",
        description="When query has high semantic clarity (specific nouns, technical terms), "
                    "give hippocampal pathway 2 extra slots.",
        apply_fn=lambda c: _set(c, "hippocampal_boost_if_semantic", True),
        affects=["semantic_baseline"],
    ))
    strategies.append(Strategy(
        id="amygdalar_diversity", name="Force amygdalar diversity",
        category="structural",
        description="Amygdalar results must span different VAD regions. "
                    "Prevents 5 memories with identical emotional signatures.",
        apply_fn=lambda c: _set(c, "amygdalar_diversity", True),
        affects=["emotional_congruence", "cross_domain_leap"],
    ))
    strategies.append(Strategy(
        id="disable_reconsolidation", name="Disable reconsolidation",
        category="formula",
        description="Stop modifying memory emotions on retrieval. "
                    "Test if reconsolidation drift hurts or helps.",
        apply_fn=lambda c: _set(c, "reconsolidation_enabled", False),
        affects=["emotional_congruence", "mood_congruent"],
    ))

    # --- COMPOUND: Multi-param coordinated changes ---
    strategies.append(Strategy(
        id="compound_semantic_rescue", name="Compound: rescue semantic baseline",
        category="compound",
        description="Increase hippocampal ratio + boost semantic spreading + "
                    "raise involuntary threshold to reduce noise.",
        apply_fn=lambda c: _multi(c, hippocampal_ratio=0.45, spreading_edge_weighting="prefer_semantic",
                                   involuntary_fire_threshold=0.45),
        affects=["semantic_baseline"],
        risk="guarded",
    ))
    strategies.append(Strategy(
        id="compound_emotional_max", name="Compound: maximize emotional recall",
        category="compound",
        description="Lower amygdalar threshold + prefer emotional spreading + "
                    "lower involuntary threshold for more Proust effects.",
        apply_fn=lambda c: _multi(c, amygdalar_emo_threshold=0.55,
                                   spreading_edge_weighting="prefer_emotional",
                                   involuntary_fire_threshold=0.25),
        affects=["emotional_congruence", "mood_congruent", "cross_domain_leap"],
        risk="guarded",
    ))
    strategies.append(Strategy(
        id="compound_cross_domain", name="Compound: maximize cross-domain leaps",
        category="compound",
        description="Lower emotional edge threshold + prefer emotional spreading + "
                    "enable emotional outlier involuntary mode.",
        apply_fn=lambda c: _multi(c, emotional_edge_threshold=0.60,
                                   spreading_edge_weighting="prefer_emotional",
                                   involuntary_mode="emotional_outlier",
                                   emotional_semantic_filter=0.65),
        affects=["cross_domain_leap"],
        risk="guarded",
    ))
    strategies.append(Strategy(
        id="compound_balanced", name="Compound: balance semantic + emotional",
        category="compound",
        description="Adaptive merge + keyword classification + moderate params. "
                    "The 'best of both worlds' attempt.",
        apply_fn=lambda c: _multi(c, merge_strategy="adaptive", query_classifier="keyword",
                                   hippocampal_ratio=0.40, amygdalar_emo_threshold=0.65),
        affects=["semantic_baseline", "emotional_congruence", "mood_congruent"],
    ))

    # --- NUMERIC: targeted param tweaks (small library for fine-tuning) ---
    for name, lo, hi, step in [
        ("hop_decay", 0.15, 0.50, 0.05),
        ("spreading_cap", 0.40, 0.90, 0.10),
        ("hippocampal_ratio", 0.20, 0.55, 0.05),
        ("amygdalar_emo_threshold", 0.50, 0.80, 0.05),
        ("involuntary_fire_threshold", 0.20, 0.50, 0.05),
    ]:
        import numpy as np  # noqa: avoid if not available
        vals = []
        v = lo
        while v <= hi + 0.001:
            vals.append(round(v, 4))
            v += step
        for val in vals:
            strategies.append(Strategy(
                id=f"param_{name}_{val}", name=f"Set {name}={val}",
                category="numeric",
                description=f"Direct parameter set: {name} = {val}",
                apply_fn=lambda c, n=name, vv=val: _set(c, n, vv),
                affects=["all"],
            ))

    return strategies


def _set(config: RetrieverConfig, key: str, value: Any) -> RetrieverConfig:
    setattr(config, key, value)
    return config

def _multi(config: RetrieverConfig, **kwargs) -> RetrieverConfig:
    for k, v in kwargs.items():
        setattr(config, k, v)
    return config


# ============================================================================
# LEDGER
# ============================================================================

@dataclass
class LedgerEntry:
    fingerprint: str
    strategy_id: str
    outcome: str  # "improved", "no-benefit", "regression", "scope-rejected", "repaired"
    score_before: float
    score_after: float
    delta: float
    category_deltas: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reason: str = ""


class Ledger:
    def __init__(self):
        self.entries: list[LedgerEntry] = []
        self._fingerprints: set[str] = set()
        self._strategy_outcomes: dict[str, list[str]] = {}

    def is_duplicate(self, fp: str) -> bool:
        return fp in self._fingerprints

    def record(self, entry: LedgerEntry):
        self.entries.append(entry)
        self._fingerprints.add(entry.fingerprint)
        self._strategy_outcomes.setdefault(entry.strategy_id, []).append(entry.outcome)

    def strategy_tried(self, strategy_id: str) -> bool:
        return strategy_id in self._strategy_outcomes

    def strategy_failed(self, strategy_id: str) -> bool:
        outcomes = self._strategy_outcomes.get(strategy_id, [])
        return len(outcomes) > 0 and outcomes[-1] != "improved"

    def failed_strategies(self) -> set[str]:
        return {sid for sid, outcomes in self._strategy_outcomes.items() if outcomes[-1] != "improved"}

    def successful_strategies(self) -> list[str]:
        return [sid for sid, outcomes in self._strategy_outcomes.items() if "improved" in outcomes]


# ============================================================================
# PLANNER — analyzes weaknesses and selects strategies
# ============================================================================

def plan_next_strategy(
    config: RetrieverConfig,
    scores: dict,
    ledger: Ledger,
    strategies: list[Strategy],
    iteration: int,
) -> Strategy | None:
    """The PLANNER phase: analyze category weaknesses and pick the best untried strategy."""
    cats = scores.get("categories", {})

    # Rank categories by how much they're losing to RAG
    cat_ranking = sorted(cats.items(), key=lambda x: x[1]["delta"])
    weakest = cat_ranking[0] if cat_ranking else None

    # Filter out already-tried strategies (fingerprint dedup)
    failed = ledger.failed_strategies()
    available = [s for s in strategies if s.id not in failed]

    if not available:
        return None

    # Phase 1: Target the weakest category
    if weakest and weakest[1]["delta"] < 0:
        # Find strategies that affect the weakest category
        targeted = [s for s in available if weakest[0] in s.affects or "all" in s.affects]
        if targeted:
            # Prefer structural/compound over numeric, and untried over retried
            def priority(s):
                cat_bonus = {"compound": 3, "structural": 2, "algorithmic": 2, "formula": 1, "numeric": 0}
                tried_penalty = 1 if ledger.strategy_tried(s.id) else 0
                return cat_bonus.get(s.category, 0) - tried_penalty
            targeted.sort(key=priority, reverse=True)
            return targeted[0]

    # Phase 2: General exploration — try untried strategies
    untried = [s for s in available if not ledger.strategy_tried(s.id)]
    if untried:
        # Prefer higher-impact (structural/compound) strategies
        untried.sort(key=lambda s: {"compound": 3, "structural": 2, "algorithmic": 2,
                                     "formula": 1, "numeric": 0}.get(s.category, 0), reverse=True)
        return untried[0]

    # Phase 3: Random exploration from non-failed
    if available:
        return random.choice(available)

    return None


# ============================================================================
# GATES
# ============================================================================

def gate_scope(config: RetrieverConfig, ledger: Ledger) -> tuple[bool, str]:
    fp = config.fingerprint()
    if ledger.is_duplicate(fp):
        return False, f"duplicate config fingerprint {fp}"
    # Weight-sum sanity
    ws = config.amygdalar_emo_weight + config.amygdalar_act_weight
    if abs(ws - 1.0) > 0.20:
        return False, f"amygdalar weights sum {ws:.2f}"
    ws2 = config.involuntary_arousal_weight + config.involuntary_activation_weight + config.involuntary_cue_weight
    if abs(ws2 - 1.0) > 0.20:
        return False, f"involuntary weights sum {ws2:.2f}"
    return True, "ok"


def gate_behavioral(new_scores: dict, baseline: dict) -> tuple[bool, str]:
    """No category can regress more than 0.08 from baseline."""
    for cat in new_scores.get("categories", {}):
        if cat in baseline.get("categories", {}):
            new_d = new_scores["categories"][cat]["delta"]
            old_d = baseline["categories"][cat]["delta"]
            if new_d < old_d - 0.08:
                return False, f"regression {cat}: {old_d:.3f}->{new_d:.3f}"
    return True, "ok"


def gate_coherence(new_scores: dict, baseline: dict) -> tuple[bool, str]:
    if new_scores["overall_eaam"] < baseline["overall_eaam"] - 0.005:
        return False, f"overall drop: {baseline['overall_eaam']:.4f}->{new_scores['overall_eaam']:.4f}"
    return True, "ok"


# ============================================================================
# MAIN XO LOOP
# ============================================================================

def run_xo_loop(
    score_fn: Callable[[RetrieverConfig], dict],
    n_iterations: int = 100,
    initial_config: RetrieverConfig | None = None,
):
    """Run the XO-ANON-XO loop: planner → editor → gates → commit/rollback."""

    strategies = _build_strategy_library()
    ledger = Ledger()
    config = initial_config.clone() if initial_config else RetrieverConfig()
    best_config = config.clone()

    print(f"  Strategy library: {len(strategies)} strategies")
    print(f"    algorithmic: {sum(1 for s in strategies if s.category=='algorithmic')}")
    print(f"    structural:  {sum(1 for s in strategies if s.category=='structural')}")
    print(f"    formula:     {sum(1 for s in strategies if s.category=='formula')}")
    print(f"    compound:    {sum(1 for s in strategies if s.category=='compound')}")
    print(f"    numeric:     {sum(1 for s in strategies if s.category=='numeric')}")

    # Baseline
    print("\n  Running baseline...")
    baseline = score_fn(config)
    best_score = baseline["overall_eaam"]
    best_delta = baseline["delta"]
    print(f"  Baseline: EAAM={best_score:.4f} Δ={best_delta:+.4f}")
    for c, d in sorted(baseline["categories"].items()):
        print(f"    {c:25} Δ={d['delta']:+.3f}")

    print(f"\n{'='*75}")
    committed = 0
    rolled_back = 0

    for iteration in range(1, n_iterations + 1):
        # PLANNER: pick strategy
        strategy = plan_next_strategy(config, baseline, ledger, strategies, iteration)
        if strategy is None:
            print(f"\n  [{iteration:3d}] No more strategies available. Stopping.")
            break

        # EDITOR: apply strategy to get candidate config
        candidate = config.clone()
        candidate = strategy.apply_fn(candidate)
        fp = candidate.fingerprint()

        # GATE 1: Scope
        passed, reason = gate_scope(candidate, ledger)
        if not passed:
            ledger.record(LedgerEntry(fp, strategy.id, "scope-rejected",
                                       best_score, best_score, 0, reason=reason))
            _log(iteration, n_iterations, strategy, best_score, best_delta, "SCOPE", reason[:35])
            continue

        # SCORE
        scores = score_fn(candidate)
        new_score = scores["overall_eaam"]
        delta = new_score - best_score

        # GATE 2: Behavioral
        passed, reason = gate_behavioral(scores, baseline)
        if not passed:
            ledger.record(LedgerEntry(fp, strategy.id, "regression",
                                       best_score, new_score, delta, reason=reason))
            rolled_back += 1
            _log(iteration, n_iterations, strategy, best_score, best_delta, "REGRESS", reason[:35])
            continue

        # GATE 3: Coherence
        passed, reason = gate_coherence(scores, baseline)
        if not passed:
            ledger.record(LedgerEntry(fp, strategy.id, "no-benefit",
                                       best_score, new_score, delta, reason=reason))
            rolled_back += 1
            _log(iteration, n_iterations, strategy, best_score, best_delta, "NO-BEN", reason[:35])
            continue

        # COMMIT or FLAT
        if delta > 0.0005:
            config = candidate
            best_config = config.clone()
            best_score = new_score
            best_delta = scores["delta"]
            baseline = scores
            committed += 1
            cat_deltas = {c: scores["categories"][c]["delta"] for c in scores["categories"]}
            ledger.record(LedgerEntry(fp, strategy.id, "improved",
                                       best_score - delta, best_score, delta,
                                       category_deltas=cat_deltas,
                                       reason=f"+{delta:.4f}"))
            _log(iteration, n_iterations, strategy, best_score, best_delta,
                 "COMMIT", f"+{delta:.4f}")
        else:
            ledger.record(LedgerEntry(fp, strategy.id, "no-benefit",
                                       best_score, new_score, delta, reason="flat"))
            rolled_back += 1
            _log(iteration, n_iterations, strategy, best_score, best_delta, "FLAT", "")

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    print(f"\n{'='*75}")
    print("XO LOOP COMPLETE")
    print(f"{'='*75}")
    print(f"  Iterations:   {min(iteration, n_iterations)}")
    print(f"  Committed:    {committed}")
    print(f"  Rolled back:  {rolled_back}")
    print(f"  Best EAAM:    {best_score:.4f}")
    print(f"  RAG baseline: {baseline['overall_rag']:.4f}")
    print(f"  Δ vs RAG:     {best_delta:+.4f}")
    pct = (best_delta / max(baseline['overall_rag'], 0.001)) * 100
    print(f"  Improvement:  {pct:.1f}%")

    print(f"\n  Outcome distribution:")
    outcomes = {}
    for e in ledger.entries:
        outcomes[e.outcome] = outcomes.get(e.outcome, 0) + 1
    for o, c in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"    {o:20} {c}")

    print(f"\n  Committed strategies (in order):")
    for e in ledger.entries:
        if e.outcome == "improved":
            s = next((s for s in strategies if s.id == e.strategy_id), None)
            cat = s.category if s else "?"
            print(f"    [{cat:10}] {e.strategy_id:40} Δ={e.delta:+.4f}")

    print(f"\n  Final category scores:")
    final = score_fn(best_config)
    for c, d in sorted(final["categories"].items()):
        w = "EAAM" if d["delta"] > 0.01 else ("RAG" if d["delta"] < -0.01 else "TIE")
        print(f"    {c:25} RAG={d['rag']:.3f}  EAAM={d['eaam']:.3f}  Δ={d['delta']:+.3f}  {w}")
    print(f"\n    OVERALL: EAAM={final['overall_eaam']:.4f}  RAG={final['overall_rag']:.4f}  "
          f"Δ={final['delta']:+.4f}  ({(final['delta']/max(final['overall_rag'],0.001))*100:.1f}%)")

    print(f"\n  Best config (structural choices):")
    for k in ["vad_similarity_fn", "merge_strategy", "query_classifier",
              "spreading_edge_weighting", "involuntary_mode",
              "hippocampal_boost_if_semantic", "amygdalar_diversity",
              "reconsolidation_enabled"]:
        print(f"    {k}: {getattr(best_config, k)}")
    print(f"  Best config (numeric):")
    for k in ["hop_decay", "spreading_cap", "fan_out_limit", "hippocampal_ratio",
              "amygdalar_emo_threshold", "amygdalar_emo_weight",
              "involuntary_fire_threshold", "emotional_edge_threshold"]:
        print(f"    {k}: {getattr(best_config, k)}")

    return best_config, best_score, ledger


def _log(iteration, total, strategy, score, delta, status, detail):
    bar_len = int(max(0, delta) * 50)
    bar = "+" * min(bar_len, 20)
    cat_tag = strategy.category[:4].upper()
    print(f"  [{iteration:3d}/{total}] [{cat_tag:4}] {strategy.id:42} "
          f"EAAM={score:.4f} Δ={delta:+.4f} [{bar:20}] {status:7} {detail}")
