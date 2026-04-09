"""MemoryStress Benchmark — EAAM vs Baseline RAG.

Longitudinal benchmark: 1,000 sessions over 10 simulated months.
Tests what breaks in production: accumulation pressure, contradiction
resolution, degradation over time, cross-agent recall, cold start.

Dataset: https://huggingface.co/datasets/singularityjason/memorystress
Paper:   https://github.com/omega-memory/memorystress

Published baselines:
  - OMEGA v1.0:  32.7% (GPT-4o)
  - Null:         0.0%

Usage:
  python3 tests/bench_memorystress.py                    # heuristic scoring
  python3 tests/bench_memorystress.py --judge             # LLM judge scoring
  python3 tests/bench_memorystress.py --sessions 200      # limit sessions
  python3 tests/bench_memorystress.py --questions 50      # limit eval questions
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.engine.consolidator import ConsolidationEngine
from eaam.models import Edge, EdgeType
from eaam.store.memory_store import MemoryStore

logger = logging.getLogger(__name__)

DATA_DIR = _ROOT / "data" / "memorystress"

DATASET_URL = (
    "https://huggingface.co/datasets/singularityjason/memorystress/"
    "resolve/main/memorystress_v1.json"
)

PHASE_BOUNDARIES = {
    1: (1, 100),
    2: (101, 500),
    3: (501, 1000),
}


@dataclass
class PhaseCheckpoint:
    phase: int
    sessions_ingested: int
    eaam_correct: int
    eaam_total: int
    rag_correct: int
    rag_total: int
    by_type: dict


def download_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fpath = DATA_DIR / "memorystress_v1.json"

    if fpath.exists():
        logger.info(f"Dataset already exists: {fpath}")
        return fpath

    print("Downloading MemoryStress dataset...")
    import httpx
    with httpx.stream("GET", DATASET_URL, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        with open(fpath, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {fpath}")
    return fpath


def load_dataset() -> dict:
    fpath = download_dataset()
    with open(fpath) as f:
        data = json.load(f)
    return data


# ============================================================================
# ANSWER EVALUATION
# ============================================================================

def heuristic_match(answer: str, expected: str) -> bool:
    """Check if retrieved text contains the expected answer."""
    if not expected or not answer:
        return False
    expected_lower = expected.lower().strip()
    answer_lower = answer.lower().strip()

    if expected_lower in answer_lower:
        return True

    expected_words = set(expected_lower.split())
    answer_words = set(answer_lower.split())
    if not expected_words:
        return False
    overlap = len(expected_words & answer_words) / len(expected_words)
    return overlap >= 0.6


def judge_match(answer: str, question: str, expected: str, judge) -> bool:
    """Use LLM judge to evaluate answer correctness."""
    result = judge.score_free_response(question, expected, answer)
    return result.score >= 50.0


# ============================================================================
# RAG BASELINE
# ============================================================================

class FlatRAGRetriever:
    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        hits = self.store.semantic_search(query, n=k)
        texts = []
        for mem_id, _ in hits:
            mem = self.store.get(mem_id)
            if mem:
                texts.append(mem.content)
        return texts


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(
    max_sessions: int = 0,
    max_questions: int = 0,
    use_judge: bool = False,
    judge_model: str | None = None,
    consolidate_every: int = 100,
):
    print("=" * 70)
    print("MemoryStress BENCHMARK: EAAM vs Flat RAG")
    print("=" * 70)

    judge = None
    if use_judge:
        from llm_judge import LLMJudge
        judge = LLMJudge(model=judge_model)
        print(f"Scoring: LLM JUDGE ({judge.provider}/{judge.model})")
    else:
        print("Scoring: HEURISTIC (keyword match)")

    data = load_dataset()
    sessions = data.get("sessions", [])
    questions = data.get("questions", [])
    facts = data.get("facts", [])
    contradictions = data.get("contradictions", data.get("contradiction_chains", []))

    if max_sessions > 0:
        sessions = sessions[:max_sessions]
    if max_questions > 0:
        questions = questions[:max_questions]

    print(f"Sessions: {len(sessions)}")
    print(f"Questions: {len(questions)}")
    print(f"Facts: {len(facts)}")
    print(f"Contradiction chains: {len(contradictions)}")
    print(f"Consolidation every {consolidate_every} sessions")

    # Load emotion model
    print("\nLoading emotion model...")
    cfg = EAAMConfig()
    cfg.emotion.use_transformer = True
    emotion_encoder = EmotionEncoder(cfg.emotion)
    print("Model loaded.\n")

    # Build EAAM store
    eaam_cfg = EAAMConfig()
    eaam_cfg.emotion.use_transformer = True
    eaam_cfg.graph.persist_path = tempfile.mkdtemp()
    eaam_cfg.vector.persist_path = tempfile.mkdtemp()

    store = MemoryStore(eaam_cfg)
    pipeline = EncodingPipeline(store, emotion_encoder, eaam_cfg)
    retriever = AssociativeRetriever(store, emotion_encoder, eaam_cfg.retrieval)
    consolidator = ConsolidationEngine(store, eaam_cfg.consolidation)
    rag = FlatRAGRetriever(store)

    def psem(memory):
        for mid, sim in pipeline.store.semantic_search(memory.content, n=10):
            if mid != memory.id and sim >= 0.30:
                pipeline.store.add_edge(Edge(
                    source_id=memory.id, target_id=mid,
                    edge_type=EdgeType.SEMANTIC, weight=sim,
                ))

    def pemo(memory):
        nbs = pipeline.store.emotional_search(memory.emotion, threshold=0.75, limit=8)
        sc = dict(pipeline.store.semantic_search(memory.content, n=20))
        for nb, es in nbs:
            if nb.id != memory.id and sc.get(nb.id, 0) < 0.50:
                pipeline.store.add_edge(Edge(
                    source_id=memory.id, target_id=nb.id,
                    edge_type=EdgeType.EMOTIONAL, weight=es,
                ))

    pipeline._build_semantic_edges = psem
    pipeline._build_emotional_edges = pemo

    # Group questions by the phase they should be asked in.
    # Each question has "phase_asked" (1, 2, or 3). We evaluate after
    # ingesting all sessions for that phase.
    questions_by_phase = defaultdict(list)
    for q in questions:
        phase = q.get("phase_asked", 1)
        questions_by_phase[phase].append(q)

    # Build checkpoint list: evaluate at the end of each phase
    checkpoints = []
    for phase in sorted(PHASE_BOUNDARIES.keys()):
        if phase in questions_by_phase:
            _, end_session = PHASE_BOUNDARIES[phase]
            checkpoints.append((phase, min(end_session, len(sessions))))

    # ======================================================================
    # INGESTION PHASE
    # ======================================================================
    print("=" * 70)
    print("INGESTION PHASE")
    print("=" * 70)

    results_by_type = defaultdict(lambda: {"eaam": 0, "rag": 0, "total": 0})
    results_by_phase = []
    results_by_age = defaultdict(lambda: {"eaam": 0, "rag": 0, "total": 0})
    contradiction_results = {"eaam": 0, "rag": 0, "total": 0}
    all_question_results = []

    session_idx = 0
    total_messages = 0
    t_start = time.time()

    for phase, target_session in checkpoints:
        # Ingest sessions up to this phase boundary
        while session_idx < target_session:
            session = sessions[session_idx]
            turns = session.get("turns", session.get("messages", []))
            sid = session.get("session_id", f"session_{session_idx}")
            agent_id = session.get("agent_id", "default")

            for turn in turns:
                content = turn.get("content", "")
                role = turn.get("role", "user")
                if content.strip():
                    pipeline.encode(
                        content=content,
                        conversation_id=f"{agent_id}_{sid}",
                        role=role,
                    )
                    total_messages += 1

            session_idx += 1

            if session_idx % 50 == 0:
                stats = store.stats()
                elapsed = time.time() - t_start
                print(f"  [{session_idx}/{len(sessions)}] {stats['node_count']} memories, "
                      f"{stats['edge_count']} edges ({elapsed:.0f}s)")

            if session_idx % consolidate_every == 0:
                print(f"  Running consolidation at session {session_idx}...")
                consolidator.consolidate(generate_reflections=True)

        # Evaluate questions for this phase
        ckpt_questions = questions_by_phase.get(phase, [])
        if not ckpt_questions:
            continue

        current_phase = phase
        print(f"\n  Evaluating {len(ckpt_questions)} questions at phase {phase} "
              f"({session_idx} sessions ingested)...")

        ckpt_eaam = 0
        ckpt_rag = 0

        for q in ckpt_questions:
            question_text = q.get("question", "")
            expected_answer = q.get("answer", q.get("expected_answer", ""))
            qtype = q.get("question_type", "unknown")
            age_sessions = q.get("age_sessions", 0)
            if age_sessions <= 100:
                fact_age = "0-100"
            elif age_sessions <= 500:
                fact_age = "100-500"
            else:
                fact_age = "500-1000"
            is_contradiction = q.get("is_contradiction", False)

            # EAAM retrieval
            eaam_results = retriever.retrieve(query=question_text, k=10)
            eaam_texts = [r.memory.content for r in eaam_results]
            eaam_answer = "\n".join(eaam_texts[:5])

            # RAG retrieval
            rag_texts = rag.retrieve(question_text, k=10)
            rag_answer = "\n".join(rag_texts[:5])

            # Score
            if use_judge and judge:
                eaam_correct = judge_match(eaam_answer, question_text, expected_answer, judge)
                rag_correct = judge_match(rag_answer, question_text, expected_answer, judge)
            else:
                eaam_correct = heuristic_match(eaam_answer, expected_answer)
                rag_correct = heuristic_match(rag_answer, expected_answer)

            if eaam_correct:
                ckpt_eaam += 1
            if rag_correct:
                ckpt_rag += 1

            results_by_type[qtype]["eaam"] += int(eaam_correct)
            results_by_type[qtype]["rag"] += int(rag_correct)
            results_by_type[qtype]["total"] += 1

            results_by_age[str(fact_age)]["eaam"] += int(eaam_correct)
            results_by_age[str(fact_age)]["rag"] += int(rag_correct)
            results_by_age[str(fact_age)]["total"] += 1

            if is_contradiction:
                contradiction_results["eaam"] += int(eaam_correct)
                contradiction_results["rag"] += int(rag_correct)
                contradiction_results["total"] += 1

            all_question_results.append({
                "question": question_text,
                "expected": expected_answer,
                "question_type": qtype,
                "phase": current_phase,
                "sessions_ingested": session_idx,
                "eaam_correct": eaam_correct,
                "rag_correct": rag_correct,
                "is_contradiction": is_contradiction,
                "fact_age": fact_age,
            })

        ckpt_total = len(ckpt_questions)
        eaam_pct = ckpt_eaam / ckpt_total * 100 if ckpt_total else 0
        rag_pct = ckpt_rag / ckpt_total * 100 if ckpt_total else 0
        print(f"  Phase {current_phase}: EAAM={eaam_pct:.1f}% ({ckpt_eaam}/{ckpt_total}) "
              f"RAG={rag_pct:.1f}% ({ckpt_rag}/{ckpt_total})")

        results_by_phase.append(PhaseCheckpoint(
            phase=current_phase,
            sessions_ingested=session_idx,
            eaam_correct=ckpt_eaam,
            eaam_total=ckpt_total,
            rag_correct=ckpt_rag,
            rag_total=ckpt_total,
            by_type=dict(results_by_type),
        ))

    total_time = time.time() - t_start
    final_stats = store.stats()

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    print(f"\n{'='*70}")
    print("MemoryStress RESULTS")
    print(f"{'='*70}")

    total_q = len(all_question_results)
    total_eaam = sum(1 for r in all_question_results if r["eaam_correct"])
    total_rag = sum(1 for r in all_question_results if r["rag_correct"])

    eaam_acc = total_eaam / total_q * 100 if total_q else 0
    rag_acc = total_rag / total_q * 100 if total_q else 0

    print(f"  Sessions ingested: {session_idx}")
    print(f"  Total messages: {total_messages}")
    print(f"  Final store: {final_stats['node_count']} memories, {final_stats['edge_count']} edges")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Questions evaluated: {total_q}")

    method = "LLM judge" if use_judge else "heuristic"
    print(f"  Scoring: {method}")

    print(f"\n  {'System':<25} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Flat RAG':<25} {rag_acc:>9.1f}% {total_rag:>10}/{total_q}")
    print(f"  {'EAAM (ours)':<25} {eaam_acc:>9.1f}% {total_eaam:>10}/{total_q}")

    # Degradation curve
    print(f"\n  DEGRADATION CURVE:")
    phase_scores = []
    for ckpt in results_by_phase:
        e_pct = ckpt.eaam_correct / ckpt.eaam_total * 100 if ckpt.eaam_total else 0
        r_pct = ckpt.rag_correct / ckpt.rag_total * 100 if ckpt.rag_total else 0
        bar = "█" * int(e_pct / 5)
        print(f"  Phase {ckpt.phase} ({ckpt.sessions_ingested:>4} sess): "
              f"EAAM={e_pct:>5.1f}%  RAG={r_pct:>5.1f}%  {bar}")
        phase_scores.append(e_pct)

    if len(phase_scores) >= 2:
        slope = (phase_scores[-1] - phase_scores[0]) / (len(phase_scores) - 1)
        print(f"  Degradation slope: {slope:+.3f}")

    # By question type
    print(f"\n  BY QUESTION TYPE:")
    print(f"  {'Type':<30} {'EAAM':>8} {'RAG':>8} {'Count':>8}")
    print(f"  {'─'*60}")
    for qtype in sorted(results_by_type.keys()):
        r = results_by_type[qtype]
        ea = r["eaam"] / r["total"] * 100 if r["total"] else 0
        ra = r["rag"] / r["total"] * 100 if r["total"] else 0
        print(f"  {qtype:<30} {ea:>7.1f}% {ra:>7.1f}% {r['total']:>8}")

    # By fact age
    print(f"\n  BY FACT AGE:")
    print(f"  {'Age Bucket':<30} {'EAAM':>8} {'RAG':>8} {'Count':>8}")
    print(f"  {'─'*60}")
    for age in sorted(results_by_age.keys()):
        r = results_by_age[age]
        ea = r["eaam"] / r["total"] * 100 if r["total"] else 0
        ra = r["rag"] / r["total"] * 100 if r["total"] else 0
        print(f"  {age:<30} {ea:>7.1f}% {ra:>7.1f}% {r['total']:>8}")

    # Contradiction resolution
    if contradiction_results["total"] > 0:
        c_e = contradiction_results["eaam"] / contradiction_results["total"] * 100
        c_r = contradiction_results["rag"] / contradiction_results["total"] * 100
        print(f"\n  CONTRADICTION RESOLUTION:")
        print(f"  EAAM: {c_e:.1f}% ({contradiction_results['eaam']}/{contradiction_results['total']})")
        print(f"  RAG:  {c_r:.1f}% ({contradiction_results['rag']}/{contradiction_results['total']})")

    print(f"\n  Published baselines:")
    print(f"  {'OMEGA v1.0 (GPT-4o)':>25} {'32.7%':>10}")
    print(f"  {'Null baseline':>25} {'0.0%':>10}")

    # Save results
    result = {
        "benchmark": "MemoryStress",
        "method": method,
        "sessions_ingested": session_idx,
        "total_messages": total_messages,
        "total_questions": total_q,
        "eaam_accuracy": eaam_acc,
        "rag_accuracy": rag_acc,
        "total_time_seconds": total_time,
        "final_store": final_stats,
        "degradation_curve": [
            {
                "phase": c.phase,
                "sessions": c.sessions_ingested,
                "eaam_pct": c.eaam_correct / c.eaam_total * 100 if c.eaam_total else 0,
                "rag_pct": c.rag_correct / c.rag_total * 100 if c.rag_total else 0,
            }
            for c in results_by_phase
        ],
        "by_type": {t: dict(r) for t, r in results_by_type.items()},
        "by_age": {a: dict(r) for a, r in results_by_age.items()},
        "contradiction_resolution": contradiction_results,
        "questions": all_question_results,
    }

    out_path = _ROOT / "results"
    out_path.mkdir(exist_ok=True)
    suffix = "_llm" if use_judge else ""
    fname = f"memorystress{suffix}_results.json"
    with open(out_path / fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to results/{fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MemoryStress benchmark for EAAM")
    parser.add_argument("--sessions", type=int, default=0, help="Max sessions to ingest (0 = all)")
    parser.add_argument("--questions", type=int, default=0, help="Max questions (0 = all)")
    parser.add_argument("--judge", action="store_true", help="Use LLM judge for scoring")
    parser.add_argument("--judge-model", type=str, default=None, help="LLM model for judge")
    parser.add_argument("--consolidate-every", type=int, default=100,
                        help="Run consolidation every N sessions")
    args = parser.parse_args()
    run_benchmark(
        max_sessions=args.sessions,
        max_questions=args.questions,
        use_judge=args.judge,
        judge_model=args.judge_model,
        consolidate_every=args.consolidate_every,
    )
