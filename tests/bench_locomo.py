"""LoCoMo MC10 Benchmark — EAAM vs Baseline RAG.

Runs both systems against the LoCoMo MC10 dataset (1,986 multiple-choice
questions across 10 multi-session conversations).

For each question:
1. All conversation sessions are encoded into memory
2. The question is used as a retrieval query
3. Retrieved memories are matched against the 10 answer choices
4. The choice with the highest overlap with retrieved text wins
   (or an LLM judge picks the answer when --judge is enabled)

Published baselines:
  - Mem0:        66.9%
  - Letta:       74.0%
  - OpenAI:      52.9%

Usage:
  python3 tests/bench_locomo.py [--limit N] [--conversations N]
  python3 tests/bench_locomo.py --judge                  # use LLM judge
  python3 tests/bench_locomo.py --judge --judge-model gpt-4o-mini
"""

from __future__ import annotations

import json, math, os, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.models import Edge, EdgeType
from eaam.store.memory_store import MemoryStore

# Optional LLM judge — imported lazily
_judge_instance = None

def get_judge(model: str | None = None):
    global _judge_instance
    if _judge_instance is None:
        from llm_judge import LLMJudge
        _judge_instance = LLMJudge(model=model)
    return _judge_instance


# ============================================================================
# DATA LOADING
# ============================================================================

LOCOMO_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/"
    "snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/"
    "data/locomo_mc10.json"
)


def load_locomo() -> list[dict]:
    data = []
    with open(LOCOMO_PATH) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def group_by_conversation(data: list[dict]) -> dict[str, list[dict]]:
    convs = defaultdict(list)
    for d in data:
        conv_id = d["question_id"].rsplit("_q", 1)[0]
        convs[conv_id].append(d)
    return dict(convs)


# ============================================================================
# ANSWER MATCHING — pick the choice best matching retrieved text
# ============================================================================

def pick_answer(retrieved_texts: list[str], choices: list[str]) -> int:
    """Pick the answer choice with highest word overlap with retrieved text."""
    retrieved_blob = " ".join(retrieved_texts).lower()
    retrieved_words = set(retrieved_blob.split())

    best_idx = 0
    best_score = -1

    for i, choice in enumerate(choices):
        choice_words = set(choice.lower().split())
        if not choice_words:
            continue
        # Weighted: exact substring match counts more than word overlap
        substring_bonus = 1.0 if choice.lower() in retrieved_blob else 0.0
        word_overlap = len(choice_words & retrieved_words) / len(choice_words)
        score = word_overlap + substring_bonus * 2.0

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def pick_answer_llm(retrieved_texts: list[str], question: str, choices: list[str], model: str | None = None) -> int:
    """Use an LLM judge to pick the best answer from retrieved context."""
    judge = get_judge(model)
    result = judge.pick_answer(question, choices, retrieved_texts)
    return result.answer_idx


def pick_answer_from_summaries(summaries: list[str], question: str, choices: list[str]) -> int:
    """For RAG baseline: pick answer from session summaries using question + choice matching."""
    blob = " ".join(summaries).lower()
    blob_words = set(blob.split())

    best_idx = 0
    best_score = -1

    for i, choice in enumerate(choices):
        choice_words = set(choice.lower().split())
        if not choice_words:
            continue
        substring_bonus = 1.0 if choice.lower() in blob else 0.0
        word_overlap = len(choice_words & blob_words) / len(choice_words)
        score = word_overlap + substring_bonus * 2.0

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(max_conversations: int = 2, max_questions_per_conv: int = 100,
                   use_judge: bool = False, judge_model: str | None = None):
    """Run LoCoMo MC10 on both EAAM and baseline."""

    print("=" * 70)
    print("LoCoMo MC10 BENCHMARK: EAAM vs Baseline")
    if use_judge:
        judge = get_judge(judge_model)
        print(f"Answer selection: LLM JUDGE ({judge.provider}/{judge.model})")
    else:
        print("Answer selection: HEURISTIC (word overlap)")
    print("=" * 70)

    data = load_locomo()
    convs = group_by_conversation(data)
    conv_ids = sorted(convs.keys())[:max_conversations]

    print(f"Total questions: {len(data)}")
    print(f"Conversations to evaluate: {len(conv_ids)} of {len(group_by_conversation(data))}")
    print(f"Max questions per conv: {max_questions_per_conv}")

    # Load emotion model once
    print("\nLoading emotion model...")
    cfg = EAAMConfig()
    cfg.emotion.use_transformer = True
    emotion_encoder = EmotionEncoder(cfg.emotion)
    print("Model loaded.")

    # Results
    eaam_correct = 0
    rag_correct = 0
    random_correct = 0
    total = 0

    type_results = defaultdict(lambda: {"eaam": 0, "rag": 0, "total": 0})

    for conv_idx, conv_id in enumerate(conv_ids):
        questions = convs[conv_id][:max_questions_per_conv]

        print(f"\n{'─'*70}")
        print(f"Conversation {conv_idx + 1}/{len(conv_ids)}: {conv_id} ({len(questions)} questions)")

        # Get session data from first question (all questions share the same sessions)
        sample = questions[0]
        sessions = sample["haystack_sessions"]
        session_ids = sample["haystack_session_ids"]
        session_datetimes = sample["haystack_session_datetimes"]
        summaries = sample["haystack_session_summaries"]

        # Build EAAM memory store for this conversation
        print(f"  Encoding {len(sessions)} sessions into EAAM...")
        t0 = time.time()

        eaam_cfg = EAAMConfig()
        eaam_cfg.emotion.use_transformer = True
        eaam_cfg.graph.persist_path = tempfile.mkdtemp()
        eaam_cfg.vector.persist_path = tempfile.mkdtemp()

        store = MemoryStore(eaam_cfg)
        pipeline = EncodingPipeline(store, emotion_encoder, eaam_cfg)
        retriever = AssociativeRetriever(store, emotion_encoder, eaam_cfg.retrieval)

        # Patch encoding
        def psem(memory):
            for mid, sim in pipeline.store.semantic_search(memory.content, n=10):
                if mid != memory.id and sim >= 0.30:
                    pipeline.store.add_edge(Edge(source_id=memory.id, target_id=mid,
                                                  edge_type=EdgeType.SEMANTIC, weight=sim))
        def pemo(memory):
            nbs = pipeline.store.emotional_search(memory.emotion, threshold=0.75, limit=8)
            sc = dict(pipeline.store.semantic_search(memory.content, n=20))
            for nb, es in nbs:
                if nb.id != memory.id and sc.get(nb.id, 0) < 0.50:
                    pipeline.store.add_edge(Edge(source_id=memory.id, target_id=nb.id,
                                                  edge_type=EdgeType.EMOTIONAL, weight=es))
        pipeline._build_semantic_edges = psem
        pipeline._build_emotional_edges = pemo

        msg_count = 0
        for sess_idx, session in enumerate(sessions):
            sid = session_ids[sess_idx] if sess_idx < len(session_ids) else f"session_{sess_idx}"
            for msg in session:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content.strip():
                    pipeline.encode(content=content, conversation_id=sid, role=role)
                    msg_count += 1

        encode_time = time.time() - t0
        stats = store.stats()
        print(f"  Encoded: {msg_count} messages -> {stats['node_count']} memories, "
              f"{stats['edge_count']} edges ({encode_time:.1f}s)")

        # Run questions
        print(f"  Running {len(questions)} questions...")
        t0 = time.time()

        for q_idx, q in enumerate(questions):
            question = q["question"]
            choices = q["choices"]
            correct_idx = q["correct_choice_index"]
            qtype = q["question_type"]

            # EAAM: retrieve and pick answer
            eaam_results = retriever.retrieve(query=question, k=10)
            eaam_texts = [r.memory.content for r in eaam_results]

            if use_judge:
                eaam_pick = pick_answer_llm(eaam_texts, question, choices, judge_model)
                rag_pick = pick_answer_llm(summaries, question, choices, judge_model)
            else:
                eaam_pick = pick_answer(eaam_texts, choices)
                rag_pick = pick_answer_from_summaries(summaries, question, choices)

            # Random baseline
            import random
            random_pick = random.randint(0, len(choices) - 1)

            # Score
            if eaam_pick == correct_idx:
                eaam_correct += 1
                type_results[qtype]["eaam"] += 1
            if rag_pick == correct_idx:
                rag_correct += 1
                type_results[qtype]["rag"] += 1
            if random_pick == correct_idx:
                random_correct += 1
            total += 1
            type_results[qtype]["total"] += 1

            if (q_idx + 1) % 25 == 0:
                eaam_acc = eaam_correct / total * 100
                rag_acc = rag_correct / total * 100
                print(f"    [{q_idx+1}/{len(questions)}] EAAM={eaam_acc:.1f}% RAG={rag_acc:.1f}% (running)")

        query_time = time.time() - t0
        print(f"  Query time: {query_time:.1f}s ({query_time/len(questions)*1000:.0f}ms/query)")

    # ======================================================================
    # FINAL REPORT
    # ======================================================================
    print(f"\n{'='*70}")
    print("LoCoMo MC10 RESULTS")
    print(f"{'='*70}")
    print(f"  Questions evaluated: {total}")
    print(f"  Conversations: {len(conv_ids)}")

    eaam_acc = eaam_correct / total * 100 if total else 0
    rag_acc = rag_correct / total * 100 if total else 0
    random_acc = random_correct / total * 100 if total else 0

    print(f"\n  {'System':<25} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'─'*50}")
    print(f"  {'Random (10-choice)':<25} {random_acc:>9.1f}% {random_correct:>10}/{total}")
    print(f"  {'RAG (summary-based)':<25} {rag_acc:>9.1f}% {rag_correct:>10}/{total}")
    print(f"  {'EAAM (ours)':<25} {eaam_acc:>9.1f}% {eaam_correct:>10}/{total}")

    print(f"\n  Published baselines (full dataset):")
    print(f"  {'OpenAI (no memory)':<25} {'52.9%':>10}")
    print(f"  {'Mem0':<25} {'66.9%':>10}")
    print(f"  {'Letta':<25} {'74.0%':>10}")

    print(f"\n  BY QUESTION TYPE:")
    print(f"  {'Type':<25} {'EAAM':>8} {'RAG':>8} {'Count':>8}")
    print(f"  {'─'*50}")
    for qtype in sorted(type_results.keys()):
        r = type_results[qtype]
        ea = r["eaam"] / r["total"] * 100 if r["total"] else 0
        ra = r["rag"] / r["total"] * 100 if r["total"] else 0
        print(f"  {qtype:<25} {ea:>7.1f}% {ra:>7.1f}% {r['total']:>8}")

    # Save results
    method = "LLM judge" if use_judge else "heuristic (word overlap)"
    result = {
        "benchmark": "LoCoMo MC10",
        "method": method,
        "total_questions": total,
        "conversations_evaluated": len(conv_ids),
        "eaam_accuracy": eaam_acc,
        "rag_accuracy": rag_acc,
        "random_accuracy": random_acc,
        "by_type": {t: {k: v for k, v in r.items()} for t, r in type_results.items()},
    }
    suffix = "_llm" if use_judge else ""
    out_path = _ROOT / "results"
    out_path.mkdir(exist_ok=True)
    fname = f"locomo_results{suffix}.json"
    with open(out_path / fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to results/{fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=int, default=2)
    parser.add_argument("--limit", type=int, default=100, help="Max questions per conversation")
    parser.add_argument("--judge", action="store_true", help="Use LLM judge instead of heuristic")
    parser.add_argument("--judge-model", type=str, default=None, help="LLM model for judge")
    args = parser.parse_args()
    run_benchmark(
        max_conversations=args.conversations,
        max_questions_per_conv=args.limit,
        use_judge=args.judge,
        judge_model=args.judge_model,
    )
