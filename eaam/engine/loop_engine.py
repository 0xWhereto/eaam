"""EAAM Loop Engine — surface-bounded autonomous self-improvement.

Adapted from xoanonxoloop's overnight engine patterns:
- Surface-bounded mutations with declared invariants
- Fingerprinted ledger preventing duplicate attempts
- Multi-gate validation pipeline (scope → behavioral → coherence)
- Cooldown on unproductive surfaces
- Repair turns on validation failure
- Wave-based execution with self-termination

Each "surface" is a region of the EAAM parameter space (spreading,
amygdalar, involuntary, merge, encoding) with declared bounds and
invariants. The engine proposes mutations within a surface, validates
them through a multi-gate pipeline, and either commits or rolls back.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ============================================================================
# LEDGER — fingerprinted attempt tracking (xoanonxoloop pattern)
# ============================================================================

@dataclass
class LedgerEntry:
    """One attempt recorded in the ledger."""
    fingerprint: str
    surface: str
    params_before: dict
    params_after: dict
    outcome: str  # "improved", "no-benefit", "validation-failed", "scope-rejected", "repair-failed"
    score_before: float
    score_after: float
    delta: float
    category_deltas: dict
    timestamp: float = field(default_factory=time.time)
    repair_turns_used: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "surface": self.surface,
            "outcome": self.outcome,
            "delta": self.delta,
            "reason": self.reason,
            "score_after": self.score_after,
            "timestamp": self.timestamp,
        }


class Ledger:
    """Append-only ledger of all attempts with fingerprint deduplication."""

    def __init__(self):
        self.entries: list[LedgerEntry] = []
        self._fingerprints: set[str] = set()

    def fingerprint(self, surface: str, params: dict) -> str:
        """Create stable hash of a surface + parameter combination."""
        # Round floats to 4 decimal places for stable fingerprinting
        stable = {k: round(v, 4) if isinstance(v, float) else v for k, v in sorted(params.items())}
        raw = json.dumps({"surface": surface, "params": stable}, sort_keys=True)
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def is_duplicate(self, fp: str) -> bool:
        return fp in self._fingerprints

    def record(self, entry: LedgerEntry):
        self.entries.append(entry)
        self._fingerprints.add(entry.fingerprint)

    def no_retry_ideas(self, surface: str, limit: int = 5) -> list[dict]:
        """Get previously failed attempts for a surface (fed to mutation strategy)."""
        failed = [e for e in self.entries if e.surface == surface and e.outcome != "improved"]
        return [e.to_dict() for e in failed[-limit:]]

    def surface_outcomes(self, surface: str) -> list[str]:
        """Get sequence of outcomes for a surface (for cooldown detection)."""
        return [e.outcome for e in self.entries if e.surface == surface]

    def consecutive_no_benefit(self, surface: str) -> int:
        """Count consecutive non-improving outcomes from the tail."""
        outcomes = self.surface_outcomes(surface)
        count = 0
        for o in reversed(outcomes):
            if o == "improved":
                break
            count += 1
        return count


# ============================================================================
# SURFACE DEFINITIONS — loaded from surfaces.yaml
# ============================================================================

@dataclass
class SurfaceParam:
    name: str
    range_min: float
    range_max: float
    default: float
    is_int: bool = False

@dataclass
class Surface:
    name: str
    description: str
    risk: str  # "safe", "guarded", "manual"
    parameters: list[SurfaceParam]
    invariants: list[str]
    affects_categories: list[str]
    attempt_limit: int = 5
    cooldown_threshold: int = 2
    repair_turns: int = 1
    # Runtime state
    attempts: int = 0
    frozen: bool = False
    cooled: bool = False
    best_score: float = -999.0


def load_surfaces(path: str | None = None) -> dict[str, Surface]:
    """Load surface definitions from YAML."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "surfaces.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    surfaces = {}

    for name, spec in raw.get("surfaces", {}).items():
        params = []
        for pname, pspec in spec.get("parameters", {}).items():
            params.append(SurfaceParam(
                name=pname,
                range_min=pspec["range"][0],
                range_max=pspec["range"][1],
                default=pspec["default"],
                is_int=pspec.get("type") == "int",
            ))
        surfaces[name] = Surface(
            name=name,
            description=spec.get("description", ""),
            risk=spec.get("risk", "safe"),
            parameters=params,
            invariants=spec.get("invariants", []),
            affects_categories=spec.get("affects_categories", []),
            attempt_limit=defaults.get("attempt_limit", 5),
            cooldown_threshold=defaults.get("cooldown_threshold", 2),
            repair_turns=defaults.get("repair_turns", 1),
        )

    return surfaces


# ============================================================================
# MULTI-GATE VALIDATION PIPELINE
# ============================================================================

def gate_scope(surface: Surface, old_params: dict, new_params: dict, ledger: Ledger) -> tuple[bool, str]:
    """Gate 1: Scope — check bounds, invariants, and duplicates."""
    # Check parameter bounds
    for sp in surface.parameters:
        val = new_params.get(sp.name, sp.default)
        if val < sp.range_min or val > sp.range_max:
            return False, f"param {sp.name}={val} out of range [{sp.range_min}, {sp.range_max}]"
        if sp.is_int and val != int(val):
            return False, f"param {sp.name}={val} must be integer"

    # Check fingerprint deduplication
    fp = ledger.fingerprint(surface.name, new_params)
    if ledger.is_duplicate(fp):
        return False, f"duplicate fingerprint {fp}"

    # Check weight-sum invariants (soft check)
    if surface.name == "amygdalar":
        w_sum = new_params.get("amygdalar_emo_weight", 0.7) + new_params.get("amygdalar_act_weight", 0.3)
        if abs(w_sum - 1.0) > 0.15:
            return False, f"amygdalar weights sum to {w_sum:.2f}, should be ~1.0"

    if surface.name == "involuntary":
        w_sum = (new_params.get("involuntary_arousal_weight", 0.3) +
                 new_params.get("involuntary_activation_weight", 0.4) +
                 new_params.get("involuntary_cue_weight", 0.3))
        if abs(w_sum - 1.0) > 0.15:
            return False, f"involuntary weights sum to {w_sum:.2f}, should be ~1.0"

    return True, "scope passed"


def gate_behavioral(scores: dict, baseline_scores: dict, surface: Surface) -> tuple[bool, str]:
    """Gate 2: Behavioral validation — no regression on affected categories."""
    affected = surface.affects_categories
    for cat in affected:
        if cat in scores["categories"] and cat in baseline_scores["categories"]:
            new_delta = scores["categories"][cat]["delta"]
            old_delta = baseline_scores["categories"][cat]["delta"]
            # Allow up to 0.05 regression on individual categories
            if new_delta < old_delta - 0.05:
                return False, f"regression on {cat}: {old_delta:.3f} -> {new_delta:.3f}"
    return True, "no regression on affected categories"


def gate_coherence(scores: dict, baseline_scores: dict) -> tuple[bool, str]:
    """Gate 3: Coherence audit — overall score must not regress significantly."""
    new_overall = scores["overall_eaam"]
    old_overall = baseline_scores["overall_eaam"]
    if new_overall < old_overall - 0.01:
        return False, f"overall regression: {old_overall:.4f} -> {new_overall:.4f}"
    return True, "overall coherence maintained"


# ============================================================================
# MUTATION STRATEGIES (informed by ledger + surface awareness)
# ============================================================================

def propose_mutation(
    surface: Surface,
    current_params: dict,
    baseline_scores: dict,
    ledger: Ledger,
    iteration: int,
    total_iterations: int,
) -> dict:
    """Propose a parameter mutation for a surface.

    Strategy:
    1. Look at ledger for failed attempts on this surface → avoid them
    2. Look at category scores for this surface's affected categories
    3. If affected categories are weak, make targeted adjustments
    4. If strong, make small exploratory adjustments
    5. Magnitude decreases with iteration (simulated annealing)
    """
    new_params = dict(current_params)
    no_retry = ledger.no_retry_ideas(surface.name, limit=5)

    # Adaptive magnitude: large early, small late (annealing schedule)
    progress = iteration / max(total_iterations, 1)
    mag = max(0.01, 0.12 * (1.0 - progress * 0.8))

    # Analyze category health for this surface
    cats = baseline_scores.get("categories", {})
    affected_deltas = [cats[c]["delta"] for c in surface.affects_categories if c in cats]
    avg_affected_delta = sum(affected_deltas) / max(len(affected_deltas), 1) if affected_deltas else 0

    # Strategy selection based on surface health
    if avg_affected_delta < -0.05:
        # Surface's categories are losing to RAG → aggressive targeted fix
        strategy = "targeted_fix"
    elif avg_affected_delta > 0.20:
        # Already winning → small exploration only
        strategy = "exploration"
    else:
        # Mixed → moderate directed search
        strategy = "directed"

    # Select which parameters to mutate
    if strategy == "targeted_fix":
        # Mutate 2-3 params with larger magnitude
        n_mutate = min(3, len(surface.parameters))
        targets = random.sample(surface.parameters, n_mutate)
        local_mag = mag * 1.5
    elif strategy == "exploration":
        # Mutate 1 param with small magnitude
        targets = [random.choice(surface.parameters)]
        local_mag = mag * 0.5
    else:
        # Mutate 1-2 params with normal magnitude
        n_mutate = min(2, len(surface.parameters))
        targets = random.sample(surface.parameters, n_mutate)
        local_mag = mag

    for sp in targets:
        current_val = current_params.get(sp.name, sp.default)
        param_range = sp.range_max - sp.range_min

        if sp.is_int:
            delta = random.choice([-1, 0, 1])
            new_val = max(int(sp.range_min), min(int(sp.range_max), int(current_val) + delta))
        else:
            # Biased direction: if category is weak, try larger values for weights
            # and smaller values for thresholds
            bias = 0.0
            if avg_affected_delta < 0 and "weight" in sp.name:
                bias = local_mag * 0.3  # bias upward for weights
            elif avg_affected_delta < 0 and "threshold" in sp.name:
                bias = -local_mag * 0.3  # bias downward for thresholds

            delta = random.gauss(bias, local_mag * param_range)
            new_val = max(sp.range_min, min(sp.range_max, current_val + delta))
            new_val = round(new_val, 4)

        new_params[sp.name] = new_val

    return new_params


def repair_mutation(
    surface: Surface,
    failed_params: dict,
    failure_reason: str,
    current_params: dict,
) -> dict:
    """Repair a failed mutation — adjust in the opposite direction of failure."""
    repaired = dict(current_params)

    # Parse failure reason to understand what went wrong
    if "regression on" in failure_reason:
        # A category regressed — pull the changed params back toward baseline
        for sp in surface.parameters:
            if failed_params.get(sp.name) != current_params.get(sp.name):
                failed_val = failed_params[sp.name]
                base_val = current_params.get(sp.name, sp.default)
                # Move 30% back toward baseline
                if isinstance(failed_val, float):
                    repaired[sp.name] = round(base_val + 0.3 * (failed_val - base_val), 4)
                else:
                    repaired[sp.name] = base_val
    else:
        # General failure — try a smaller mutation from baseline
        for sp in surface.parameters:
            if failed_params.get(sp.name) != current_params.get(sp.name):
                base_val = current_params.get(sp.name, sp.default)
                if isinstance(base_val, float):
                    small_delta = random.gauss(0, 0.02 * (sp.range_max - sp.range_min))
                    repaired[sp.name] = round(
                        max(sp.range_min, min(sp.range_max, base_val + small_delta)), 4
                    )

    return repaired


# ============================================================================
# MAIN LOOP ENGINE
# ============================================================================

def run_loop_engine(
    score_fn,
    n_iterations: int = 100,
    surfaces_path: str | None = None,
    initial_params: dict | None = None,
):
    """Run the surface-bounded loop engine.

    Args:
        score_fn: callable(params_dict) -> scores_dict
            Must return {"overall_eaam": float, "overall_rag": float, "delta": float,
                         "categories": {cat: {"rag": float, "eaam": float, "delta": float}}}
        n_iterations: total iteration budget
        surfaces_path: path to surfaces.yaml
        initial_params: starting parameter values
    """
    surfaces = load_surfaces(surfaces_path)
    ledger = Ledger()

    # Initialize parameters from defaults
    current_params = {}
    for surface in surfaces.values():
        for sp in surface.parameters:
            current_params[sp.name] = sp.default
    if initial_params:
        current_params.update(initial_params)

    best_params = dict(current_params)

    # Run baseline
    print("  Running baseline...")
    baseline_scores = score_fn(current_params)
    best_score = baseline_scores["overall_eaam"]
    best_delta = baseline_scores["delta"]

    print(f"  Baseline: EAAM={best_score:.4f} RAG={baseline_scores['overall_rag']:.4f} "
          f"Δ={best_delta:+.4f}")
    for cat, data in sorted(baseline_scores["categories"].items()):
        print(f"    {cat:25} Δ={data['delta']:+.3f}")

    # Wave loop
    wave = 0
    total_attempts = 0
    total_improved = 0
    total_rollback = 0
    total_cooled = 0

    history: list[dict] = []

    while total_attempts < n_iterations:
        wave += 1

        # Find runnable surfaces
        runnable = [
            s for s in surfaces.values()
            if not s.frozen and not s.cooled and s.attempts < s.attempt_limit
        ]
        if not runnable:
            # Reset cooldowns and try again with increased attempt limits
            any_reset = False
            for s in surfaces.values():
                if s.cooled:
                    s.cooled = False
                    s.attempts = 0
                    any_reset = True
            if not any_reset:
                print(f"\n  All surfaces exhausted at wave {wave}. Stopping.")
                break
            print(f"\n  --- Wave {wave}: Reset cooled surfaces ---")
            continue

        # Sort surfaces: worst-performing first (focus effort where needed)
        def surface_health(s):
            cats = baseline_scores.get("categories", {})
            deltas = [cats[c]["delta"] for c in s.affects_categories if c in cats]
            return sum(deltas) / max(len(deltas), 1) if deltas else 0
        runnable.sort(key=surface_health)

        for surface in runnable:
            if total_attempts >= n_iterations:
                break

            total_attempts += 1
            surface.attempts += 1

            # Extract current surface params
            surface_param_names = {sp.name for sp in surface.parameters}
            surface_current = {k: v for k, v in current_params.items() if k in surface_param_names}

            # Propose mutation
            proposed = propose_mutation(
                surface, surface_current, baseline_scores, ledger,
                total_attempts, n_iterations,
            )

            # Gate 1: Scope
            full_proposed = dict(current_params)
            full_proposed.update(proposed)
            passed, reason = gate_scope(surface, surface_current, proposed, ledger)
            if not passed:
                fp = ledger.fingerprint(surface.name, proposed)
                ledger.record(LedgerEntry(
                    fingerprint=fp, surface=surface.name,
                    params_before=surface_current, params_after=proposed,
                    outcome="scope-rejected", score_before=best_score, score_after=best_score,
                    delta=0.0, category_deltas={}, reason=reason,
                ))
                _log(total_attempts, n_iterations, surface.name, best_score, best_delta, "SCOPE", reason[:40])
                continue

            # Score with proposed params
            scores = score_fn(full_proposed)
            fp = ledger.fingerprint(surface.name, proposed)

            # Gate 2: Behavioral (no regression on affected categories)
            passed, reason = gate_behavioral(scores, baseline_scores, surface)
            if not passed:
                # Repair turn
                repaired = None
                for repair in range(surface.repair_turns):
                    repaired_params = repair_mutation(surface, proposed, reason, surface_current)
                    full_repaired = dict(current_params)
                    full_repaired.update(repaired_params)
                    repair_scores = score_fn(full_repaired)
                    r_passed, r_reason = gate_behavioral(repair_scores, baseline_scores, surface)
                    if r_passed:
                        scores = repair_scores
                        proposed = repaired_params
                        full_proposed = full_repaired
                        fp = ledger.fingerprint(surface.name, proposed)
                        reason = f"repaired: {r_reason}"
                        passed = True
                        break

                if not passed:
                    ledger.record(LedgerEntry(
                        fingerprint=fp, surface=surface.name,
                        params_before=surface_current, params_after=proposed,
                        outcome="validation-failed", score_before=best_score,
                        score_after=scores["overall_eaam"],
                        delta=scores["overall_eaam"] - best_score,
                        category_deltas={c: scores["categories"].get(c, {}).get("delta", 0)
                                         for c in surface.affects_categories},
                        reason=reason,
                    ))
                    total_rollback += 1
                    _log(total_attempts, n_iterations, surface.name, best_score, best_delta, "REGRESS", reason[:40])

                    # Check cooldown
                    if ledger.consecutive_no_benefit(surface.name) >= surface.cooldown_threshold:
                        surface.cooled = True
                        total_cooled += 1
                        _log(total_attempts, n_iterations, surface.name, best_score, best_delta, "COOLED", "")
                    continue

            # Gate 3: Coherence (overall must not regress)
            passed, reason = gate_coherence(scores, baseline_scores)
            if not passed:
                ledger.record(LedgerEntry(
                    fingerprint=fp, surface=surface.name,
                    params_before=surface_current, params_after=proposed,
                    outcome="no-benefit", score_before=best_score,
                    score_after=scores["overall_eaam"],
                    delta=scores["overall_eaam"] - best_score,
                    category_deltas={}, reason=reason,
                ))
                total_rollback += 1
                _log(total_attempts, n_iterations, surface.name, best_score, best_delta, "NO-BEN", reason[:40])

                if ledger.consecutive_no_benefit(surface.name) >= surface.cooldown_threshold:
                    surface.cooled = True
                    total_cooled += 1
                continue

            # All gates passed — commit
            new_score = scores["overall_eaam"]
            new_delta = scores["delta"]
            improvement = new_score - best_score

            if improvement > 0.0001:
                # Actual improvement — commit and freeze surface
                ledger.record(LedgerEntry(
                    fingerprint=fp, surface=surface.name,
                    params_before=surface_current, params_after=proposed,
                    outcome="improved", score_before=best_score,
                    score_after=new_score,
                    delta=improvement,
                    category_deltas={c: scores["categories"].get(c, {}).get("delta", 0)
                                     for c in surface.affects_categories},
                    reason=f"improved by {improvement:+.4f}",
                ))
                current_params.update(proposed)
                best_params = dict(current_params)
                best_score = new_score
                best_delta = new_delta
                baseline_scores = scores  # update baseline
                surface.frozen = True  # freeze after success
                total_improved += 1

                _log(total_attempts, n_iterations, surface.name, best_score, best_delta,
                     "COMMIT", f"+{improvement:.4f}")

                history.append({
                    "iteration": total_attempts, "score": best_score,
                    "delta": best_delta, "surface": surface.name,
                })
            else:
                # Passed gates but no meaningful improvement
                ledger.record(LedgerEntry(
                    fingerprint=fp, surface=surface.name,
                    params_before=surface_current, params_after=proposed,
                    outcome="no-benefit", score_before=best_score,
                    score_after=new_score, delta=improvement,
                    category_deltas={}, reason="gates passed but no meaningful improvement",
                ))
                total_rollback += 1
                _log(total_attempts, n_iterations, surface.name, best_score, best_delta, "FLAT", "")

                if ledger.consecutive_no_benefit(surface.name) >= surface.cooldown_threshold:
                    surface.cooled = True
                    total_cooled += 1

    # Final report
    print(f"\n{'='*70}")
    print("LOOP ENGINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Total attempts:    {total_attempts}")
    print(f"  Improvements:      {total_improved}")
    print(f"  Rollbacks:         {total_rollback}")
    print(f"  Surfaces cooled:   {total_cooled}")
    print(f"  Best EAAM:         {best_score:.4f}")
    print(f"  RAG baseline:      {baseline_scores['overall_rag']:.4f}")
    print(f"  Best delta vs RAG: {best_delta:+.4f}")
    improvement_pct = (best_delta / max(baseline_scores['overall_rag'], 0.001)) * 100
    print(f"  Improvement:       {improvement_pct:.1f}%")

    print(f"\n  Ledger summary:")
    outcomes = {}
    for e in ledger.entries:
        outcomes[e.outcome] = outcomes.get(e.outcome, 0) + 1
    for outcome, count in sorted(outcomes.items()):
        print(f"    {outcome}: {count}")

    print(f"\n  Surface states:")
    for name, s in surfaces.items():
        state = "FROZEN" if s.frozen else ("COOLED" if s.cooled else "active")
        print(f"    {name:15} attempts={s.attempts} state={state}")

    print(f"\n  Best parameters:")
    for k, v in sorted(best_params.items()):
        print(f"    {k}: {v}")

    if history:
        print(f"\n  Improvement timeline:")
        for h in history:
            print(f"    [{h['iteration']:3d}] {h['surface']:15} score={h['score']:.4f} Δ={h['delta']:+.4f}")

    return best_params, best_score, ledger


def _log(attempt, total, surface, score, delta, status, reason):
    bar = "+" * int(max(0, delta) * 40) + "-" * int(max(0, -delta) * 40)
    print(f"  [{attempt:3d}/{total}] {surface:15} EAAM={score:.4f} Δ={delta:+.4f} "
          f"[{bar[:15]:15}] {status:7} {reason}")
