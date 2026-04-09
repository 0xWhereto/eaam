"""LongMemEval Benchmark — EAAM vs Baseline RAG.

Tests 5 core long-term memory abilities from the ICLR 2025 benchmark:
  1. Information Extraction  — recall specific details from history
  2. Multi-Session Reasoning — synthesize across sessions
  3. Knowledge Updates       — track changed user info over time
  4. Temporal Reasoning      — use timestamps and time references
  5. Abstention              — refuse to answer unknown questions

Dataset: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Paper:   https://arxiv.org/abs/2410.10813

Published baselines (LongMemEval_S, GPT-4o reader):
  Full history (128k context): 46.8%
  Flat BM25 retrieval:         32.4%
  Flat Stella retrieval:       37.2%

Usage:
  python3 tests/bench_longmemeval.py                    # heuristic scoring
  python3 tests/bench_longmemeval.py --judge             # LLM judge scoring
  python3 tests/bench_longmemeval.py --limit 50          # first 50 questions
  python3 tests/bench_longmemeval.py --dataset oracle    # oracle (evidence-only)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TESTS))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.models import Edge, EdgeType
from eaam.store.memory_store import MemoryStore

logger = logging.getLogger(__name__)

DATA_DIR = _ROOT / "data" / "longmemeval"

DATASET_URLS = {
    "oracle": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
}

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
    "abstention",
]

# Maps raw question_type values to our unified categories
TYPE_MAP = {
    "single-session-user": "information_extraction",
    "single-session-assistant": "information_extraction",
    "single-session-preference": "information_extraction",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
    "multi-session": "multi_session_reasoning",
}


def download_dataset(variant: str = "s") -> Path:
    url = DATASET_URLS.get(variant)
    if not url:
        raise ValueError(f"Unknown dataset variant: {variant}. Use 'oracle' or 's'.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"longmemeval_{variant}.json"
    fpath = DATA_DIR / fname

    if fpath.exists():
        logger.info(f"Dataset already exists: {fpath}")
        return fpath

    print(f"Downloading LongMemEval ({variant})...")
    import httpx
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        with open(fpath, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {fpath}")
    return fpath


def load_dataset(variant: str = "s") -> list[dict]:
    fpath = download_dataset(variant)
    with open(fpath) as f:
        data = json.load(f)
    return data


def classify_question(item: dict) -> str:
    qid = item.get("question_id", "")
    if qid.endswith("_abs"):
        return "abstention"
    return TYPE_MAP.get(item.get("question_type", ""), "other")


# ============================================================================
# ANSWER MATCHING
# ============================================================================

def heuristic_answer(retrieved_texts: list[str], question: str, expected: str) -> tuple[str, float]:
    """Generate answer from retrieved texts using keyword overlap scoring."""
    if not retrieved_texts:
        return "I don't have enough information to answer this question.", 0.0

    blob = " ".join(retrieved_texts).lower()
    expected_words = set(expected.lower().split())
    blob_words = set(blob.split())

    if not expected_words:
        return retrieved_texts[0][:500], 0.0

    overlap = len(expected_words & blob_words) / len(expected_words)
    best_text = retrieved_texts[0]

    for text in retrieved_texts:
        text_words = set(text.lower().split())
        score = len(expected_words & text_words) / len(expected_words)
        if score > overlap:
            overlap = score
            best_text = text

    return best_text[:500], overlap


def judge_answer(
    retrieved_texts: list[str],
    question: str,
    expected: str,
    judge,
) -> tuple[str, float]:
    """Use LLM judge to generate and score an answer."""
    context = "\n".join(f"[Memory {i+1}] {t}" for i, t in enumerate(retrieved_texts[:10]))
    prompt = (
        f"Based on the following conversation memories, answer the question.\n\n"
        f"## Memories\n{context}\n\n"
        f"## Question\n{question}\n\n"
        f"Answer concisely based only on the memories above. "
        f"If the memories don't contain enough information, say so."
    )
    answer = judge._call_llm(prompt)
    result = judge.score_free_response(question, expected, answer)
    return answer, result.score / 100.0


# ============================================================================
# RAG BASELINE
# ============================================================================

class FlatRAGRetriever:
    """Simple vector-only retriever for baseline comparison."""

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
# SESSION RETRIEVAL ACCURACY
# ============================================================================

def session_recall(retrieved_conv_ids: set[str], answer_session_ids: list[str]) -> float:
    if not answer_session_ids:
        return 0.0
    found = sum(1 for sid in answer_session_ids if sid in retrieved_conv_ids)
    return found / len(answer_session_ids)


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(
    dataset_variant: str = "s",
    max_questions: int = 0,
    use_judge: bool = False,
    judge_model: str | None = None,
):
    print("=" * 70)
    print("LongMemEval BENCHMARK: EAAM vs Flat RAG")
    print("=" * 70)

    judge = None
    if use_judge:
        from llm_judge import LLMJudge
        judge = LLMJudge(model=judge_model)
        print(f"Scoring: LLM JUDGE ({judge.provider}/{judge.model})")
    else:
        print("Scoring: HEURISTIC (keyword overlap)")

    data = load_dataset(dataset_variant)
    if max_questions > 0:
        data = data[:max_questions]

    print(f"Dataset: LongMemEval_{dataset_variant} ({len(data)} questions)")

    # Load emotion model once
    print("\nLoading emotion model...")
    cfg = EAAMConfig()
    cfg.emotion.use_transformer = True
    emotion_encoder = EmotionEncoder(cfg.emotion)
    print("Model loaded.\n")

    # Group questions by their haystack to avoid re-encoding
    # In LongMemEval, each question has its own haystack, but many share sessions
    # For efficiency, we process each question independently but reuse the emotion model

    results_by_type = defaultdict(lambda: {"eaam_scores": [], "rag_scores": [], "session_recalls": []})
    all_eaam_scores = []
    all_rag_scores = []
    all_session_recalls = []
    hypotheses = []

    for q_idx, item in enumerate(data):
        question = item["question"]
        expected = item["answer"]
        qtype = classify_question(item)
        sessions = item["haystack_sessions"]
        session_ids = item.get("haystack_session_ids", [])
        answer_session_ids = item.get("answer_session_ids", [])

        if (q_idx + 1) % 10 == 0 or q_idx == 0:
            print(f"[{q_idx+1}/{len(data)}] Encoding {len(sessions)} sessions...")

        # Build fresh EAAM store per question
        eaam_cfg = EAAMConfig()
        eaam_cfg.emotion.use_transformer = True
        eaam_cfg.graph.persist_path = tempfile.mkdtemp()
        eaam_cfg.vector.persist_path = tempfile.mkdtemp()

        store = MemoryStore(eaam_cfg)
        pipeline = EncodingPipeline(store, emotion_encoder, eaam_cfg)
        retriever = AssociativeRetriever(store, emotion_encoder, eaam_cfg.retrieval)
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

        # Encode all sessions
        for sess_idx, session in enumerate(sessions):
            sid = session_ids[sess_idx] if sess_idx < len(session_ids) else f"session_{sess_idx}"
            for msg in session:
                content = msg.get("content", "")
                role = msg.get("role", "user")
                if content.strip():
                    pipeline.encode(content=content, conversation_id=sid, role=role)

        # EAAM retrieval
        eaam_results = retriever.retrieve(query=question, k=10)
        eaam_texts = [r.memory.content for r in eaam_results]
        eaam_conv_ids = {r.memory.conversation_id for r in eaam_results}

        # RAG retrieval
        rag_texts = rag.retrieve(question, k=10)

        # Score
        if use_judge and judge:
            eaam_answer, eaam_score = judge_answer(eaam_texts, question, expected, judge)
            rag_answer, rag_score = judge_answer(rag_texts, question, expected, judge)
        else:
            eaam_answer, eaam_score = heuristic_answer(eaam_texts, question, expected)
            rag_answer, rag_score = heuristic_answer(rag_texts, question, expected)

        # Session-level recall (EAAM only)
        s_recall = session_recall(eaam_conv_ids, answer_session_ids)

        # Abstention handling: for abstention questions, the correct behavior is
        # to say "I don't know" — penalize confident wrong answers
        if qtype == "abstention":
            abstention_keywords = {"don't know", "no information", "not mentioned",
                                   "cannot answer", "no relevant", "not enough"}
            eaam_lower = eaam_answer.lower()
            if any(kw in eaam_lower for kw in abstention_keywords):
                eaam_score = 1.0
            else:
                eaam_score = 0.0

            rag_lower = (rag_answer if isinstance(rag_answer, str) else "").lower()
            if any(kw in rag_lower for kw in abstention_keywords):
                rag_score = 1.0
            else:
                rag_score = 0.0

        results_by_type[qtype]["eaam_scores"].append(eaam_score)
        results_by_type[qtype]["rag_scores"].append(rag_score)
        results_by_type[qtype]["session_recalls"].append(s_recall)
        all_eaam_scores.append(eaam_score)
        all_rag_scores.append(rag_score)
        all_session_recalls.append(s_recall)

        hypotheses.append({
            "question_id": item.get("question_id", f"q_{q_idx}"),
            "question_type": qtype,
            "hypothesis": eaam_answer[:500] if isinstance(eaam_answer, str) else "",
            "eaam_score": eaam_score,
            "rag_score": rag_score,
            "session_recall": s_recall,
        })

        if (q_idx + 1) % 25 == 0:
            avg_e = sum(all_eaam_scores) / len(all_eaam_scores) * 100
            avg_r = sum(all_rag_scores) / len(all_rag_scores) * 100
            print(f"  [{q_idx+1}] Running — EAAM={avg_e:.1f}% RAG={avg_r:.1f}%")

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    print(f"\n{'='*70}")
    print("LongMemEval RESULTS")
    print(f"{'='*70}")
    print(f"  Questions evaluated: {len(data)}")
    print(f"  Dataset: LongMemEval_{dataset_variant}")
    method = "LLM judge" if use_judge else "heuristic"
    print(f"  Scoring: {method}")

    overall_eaam = sum(all_eaam_scores) / len(all_eaam_scores) * 100 if all_eaam_scores else 0
    overall_rag = sum(all_rag_scores) / len(all_rag_scores) * 100 if all_rag_scores else 0
    overall_recall = sum(all_session_recalls) / len(all_session_recalls) * 100 if all_session_recalls else 0

    print(f"\n  {'System':<25} {'Score':>10}")
    print(f"  {'─'*40}")
    print(f"  {'Flat RAG':<25} {overall_rag:>9.1f}%")
    print(f"  {'EAAM (ours)':<25} {overall_eaam:>9.1f}%")
    print(f"  {'EAAM session recall':<25} {overall_recall:>9.1f}%")

    print(f"\n  BY CATEGORY:")
    print(f"  {'Category':<25} {'EAAM':>8} {'RAG':>8} {'Recall':>8} {'Count':>8}")
    print(f"  {'─'*60}")
    for qtype in sorted(results_by_type.keys()):
        r = results_by_type[qtype]
        n = len(r["eaam_scores"])
        avg_e = sum(r["eaam_scores"]) / n * 100 if n else 0
        avg_r = sum(r["rag_scores"]) / n * 100 if n else 0
        avg_s = sum(r["session_recalls"]) / n * 100 if n else 0
        print(f"  {qtype:<25} {avg_e:>7.1f}% {avg_r:>7.1f}% {avg_s:>7.1f}% {n:>8}")

    print(f"\n  Published baselines (LongMemEval_S):")
    print(f"  {'Full history (128k)':>25} {'46.8%':>10}")
    print(f"  {'Flat BM25':>25} {'32.4%':>10}")
    print(f"  {'Flat Stella':>25} {'37.2%':>10}")

    # Save results
    result = {
        "benchmark": "LongMemEval",
        "dataset": f"longmemeval_{dataset_variant}",
        "method": method,
        "total_questions": len(data),
        "overall_eaam_score": overall_eaam,
        "overall_rag_score": overall_rag,
        "overall_session_recall": overall_recall,
        "by_type": {
            qtype: {
                "eaam_avg": sum(r["eaam_scores"]) / len(r["eaam_scores"]) * 100 if r["eaam_scores"] else 0,
                "rag_avg": sum(r["rag_scores"]) / len(r["rag_scores"]) * 100 if r["rag_scores"] else 0,
                "session_recall": sum(r["session_recalls"]) / len(r["session_recalls"]) * 100 if r["session_recalls"] else 0,
                "count": len(r["eaam_scores"]),
            }
            for qtype, r in results_by_type.items()
        },
        "hypotheses": hypotheses,
    }

    out_path = _ROOT / "results"
    out_path.mkdir(exist_ok=True)
    suffix = "_llm" if use_judge else ""
    fname = f"longmemeval_{dataset_variant}{suffix}_results.json"
    with open(out_path / fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to results/{fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LongMemEval benchmark for EAAM")
    parser.add_argument("--dataset", choices=["oracle", "s"], default="s",
                        help="Dataset variant: 'oracle' (evidence only) or 's' (~115k tokens)")
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    parser.add_argument("--judge", action="store_true", help="Use LLM judge for scoring")
    parser.add_argument("--judge-model", type=str, default=None, help="LLM model for judge")
    args = parser.parse_args()
    run_benchmark(
        dataset_variant=args.dataset,
        max_questions=args.limit,
        use_judge=args.judge,
        judge_model=args.judge_model,
    )
