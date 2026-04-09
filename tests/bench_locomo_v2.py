"""LoCoMo MC10 Benchmark v2 — EAAM with embedding-based answer selection.

Instead of simple word overlap, uses sentence-transformer embeddings to:
1. Encode the question + each answer choice as a combined query
2. Score each choice by how well EAAM's retrieved context matches
3. Pick the choice with highest retrieval-context alignment

This approximates LLM reasoning by using semantic understanding of
whether retrieved memories support each answer choice.

Published baselines:
  - Random:      10.0%
  - OpenAI:      52.9% (LLM reasoning, no memory)
  - Mem0:        66.9% (LLM + memory)
  - Letta:       74.0% (LLM + filesystem)

Usage: python3 tests/bench_locomo_v2.py [--conversations N] [--limit N]
"""

from __future__ import annotations

import json, os, sys, tempfile, time
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

LOCOMO_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--Percena--locomo-mc10/"
    "snapshots/7d59a0463d83f97b042684310c0b3d17553004cd/"
    "data/locomo_mc10.json"
)


def load_locomo():
    data = []
    with open(LOCOMO_PATH) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def group_by_conversation(data):
    convs = defaultdict(list)
    for d in data:
        conv_id = d["question_id"].rsplit("_q", 1)[0]
        convs[conv_id].append(d)
    return dict(convs)


class EmbeddingScorer:
    """Score answer choices by embedding similarity with retrieved context."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print("  Loading embedding scorer...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def pick_answer(self, question: str, choices: list[str],
                    retrieved_texts: list[str], summaries: list[str] = None) -> int:
        """Pick the answer choice best supported by retrieved context.

        Strategy:
        1. Build a context blob from retrieved memories
        2. For each choice, compute similarity(question+choice, context)
        3. Also check if choice text appears as substring in context
        4. Pick highest combined score
        """
        if not retrieved_texts:
            # Fallback to summaries if no retrieval
            context = " ".join(summaries or [])
        else:
            context = " ".join(retrieved_texts)

        if not context.strip():
            return 0

        # Encode context as one blob
        context_emb = self.model.encode([context], show_progress_bar=False)[0]

        best_idx = 0
        best_score = -1

        for i, choice in enumerate(choices):
            # Combine question + choice for richer embedding
            combined = f"{question} {choice}"
            choice_emb = self.model.encode([combined], show_progress_bar=False)[0]

            # Cosine similarity
            dot = sum(a * b for a, b in zip(choice_emb, context_emb))
            norm_a = sum(a ** 2 for a in choice_emb) ** 0.5
            norm_b = sum(b ** 2 for b in context_emb) ** 0.5
            cosine_sim = dot / (norm_a * norm_b) if norm_a and norm_b else 0

            # Substring bonus (exact match in context is strong signal)
            substring_bonus = 0.3 if choice.lower() in context.lower() else 0.0

            # Word overlap bonus
            choice_words = set(choice.lower().split())
            context_words = set(context.lower().split())
            overlap = len(choice_words & context_words) / max(len(choice_words), 1)
            word_bonus = overlap * 0.2

            score = cosine_sim + substring_bonus + word_bonus

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx


def run_benchmark(max_conversations: int = 3, max_questions: int = 150):
    print("=" * 70)
    print("LoCoMo MC10 v2: EAAM + Embedding Reasoning")
    print("=" * 70)

    data = load_locomo()
    convs = group_by_conversation(data)
    conv_ids = sorted(convs.keys())[:max_conversations]

    print(f"Total dataset: {len(data)} questions, {len(convs)} conversations")
    print(f"Evaluating: {len(conv_ids)} conversations, up to {max_questions} questions each")

    # Load models
    print("\nLoading models...")
    cfg = EAAMConfig()
    cfg.emotion.use_transformer = True
    emotion_encoder = EmotionEncoder(cfg.emotion)
    scorer = EmbeddingScorer()
    print("Models loaded.\n")

    # Results tracking
    eaam_correct = 0
    rag_correct = 0  # RAG = summaries + embedding scoring
    baseline_correct = 0  # just summaries + word overlap
    total = 0
    type_results = defaultdict(lambda: {"eaam": 0, "rag": 0, "baseline": 0, "total": 0})

    for conv_idx, conv_id in enumerate(conv_ids):
        questions = convs[conv_id][:max_questions]

        print(f"{'─'*70}")
        print(f"Conversation {conv_idx+1}/{len(conv_ids)}: {conv_id} ({len(questions)} questions)")

        sample = questions[0]
        sessions = sample["haystack_sessions"]
        session_ids = sample["haystack_session_ids"]
        summaries = sample["haystack_session_summaries"]

        # Build EAAM memory
        print(f"  Encoding {len(sessions)} sessions...")
        t0 = time.time()

        eaam_cfg = EAAMConfig()
        eaam_cfg.emotion.use_transformer = True
        eaam_cfg.graph.persist_path = tempfile.mkdtemp()
        eaam_cfg.vector.persist_path = tempfile.mkdtemp()

        store = MemoryStore(eaam_cfg)
        pipeline = EncodingPipeline(store, emotion_encoder, eaam_cfg)
        retriever = AssociativeRetriever(store, emotion_encoder, eaam_cfg.retrieval)

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
            sid = session_ids[sess_idx] if sess_idx < len(session_ids) else f"s_{sess_idx}"
            for msg in session:
                content = msg.get("content", "").strip()
                if content:
                    pipeline.encode(content=content, conversation_id=sid, role=msg.get("role", "user"))
                    msg_count += 1

        encode_time = time.time() - t0
        stats = store.stats()
        print(f"  Encoded: {msg_count} msgs -> {stats['node_count']} memories, {stats['edge_count']} edges ({encode_time:.1f}s)")

        # Run questions
        print(f"  Evaluating {len(questions)} questions...")
        t0 = time.time()

        for q_idx, q in enumerate(questions):
            question = q["question"]
            choices = q["choices"]
            correct_idx = q["correct_choice_index"]
            qtype = q["question_type"]

            # EAAM: retrieve + embedding-based answer selection
            eaam_results = retriever.retrieve(query=question, k=10)
            eaam_texts = [r.memory.content for r in eaam_results]
            eaam_pick = scorer.pick_answer(question, choices, eaam_texts)

            # RAG baseline: summaries + embedding-based answer selection
            rag_pick = scorer.pick_answer(question, choices, [], summaries)

            # Simple baseline: summaries + word overlap only
            summary_blob = " ".join(summaries).lower()
            baseline_best = 0
            baseline_best_score = -1
            for i, choice in enumerate(choices):
                cw = set(choice.lower().split())
                sw = set(summary_blob.split())
                overlap = len(cw & sw) / max(len(cw), 1)
                bonus = 1.0 if choice.lower() in summary_blob else 0.0
                s = overlap + bonus * 2
                if s > baseline_best_score:
                    baseline_best_score = s
                    baseline_best = i

            if eaam_pick == correct_idx:
                eaam_correct += 1
                type_results[qtype]["eaam"] += 1
            if rag_pick == correct_idx:
                rag_correct += 1
                type_results[qtype]["rag"] += 1
            if baseline_best == correct_idx:
                baseline_correct += 1
                type_results[qtype]["baseline"] += 1
            total += 1
            type_results[qtype]["total"] += 1

            if (q_idx + 1) % 25 == 0:
                ea = eaam_correct / total * 100
                ra = rag_correct / total * 100
                print(f"    [{q_idx+1}/{len(questions)}] EAAM={ea:.1f}% RAG={ra:.1f}%")

        qtime = time.time() - t0
        print(f"  Query time: {qtime:.1f}s ({qtime/len(questions)*1000:.0f}ms/q)")

    # ── REPORT ──
    print(f"\n{'='*70}")
    print("LoCoMo MC10 v2 RESULTS")
    print(f"{'='*70}")
    print(f"  Questions: {total}")
    print(f"  Conversations: {len(conv_ids)}")

    ea = eaam_correct / total * 100 if total else 0
    ra = rag_correct / total * 100 if total else 0
    ba = baseline_correct / total * 100 if total else 0

    print(f"\n  {'System':<35} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'─'*60}")
    print(f"  {'Random (10-choice)':<35} {'10.0%':>10}")
    print(f"  {'Word overlap (summaries)':<35} {ba:>9.1f}% {baseline_correct:>10}/{total}")
    print(f"  {'RAG + Embedding scorer':<35} {ra:>9.1f}% {rag_correct:>10}/{total}")
    print(f"  {'EAAM + Embedding scorer':<35} {ea:>9.1f}% {eaam_correct:>10}/{total}")
    print(f"\n  Published baselines (full dataset, with LLM reasoning):")
    print(f"  {'OpenAI (no memory + LLM)':<35} {'52.9%':>10}")
    print(f"  {'Mem0 (memory + LLM)':<35} {'66.9%':>10}")
    print(f"  {'Letta (filesystem + LLM)':<35} {'74.0%':>10}")

    print(f"\n  BY QUESTION TYPE:")
    print(f"  {'Type':<25} {'EAAM':>8} {'RAG':>8} {'Base':>8} {'Count':>8}")
    print(f"  {'─'*55}")
    for qtype in sorted(type_results.keys()):
        r = type_results[qtype]
        e = r["eaam"] / r["total"] * 100 if r["total"] else 0
        rg = r["rag"] / r["total"] * 100 if r["total"] else 0
        b = r["baseline"] / r["total"] * 100 if r["total"] else 0
        print(f"  {qtype:<25} {e:>7.1f}% {rg:>7.1f}% {b:>7.1f}% {r['total']:>8}")

    # Save
    result = {
        "benchmark": "LoCoMo MC10 v2",
        "method": "EAAM retrieval + embedding scoring (no LLM)",
        "total_questions": total,
        "conversations": len(conv_ids),
        "eaam_accuracy": round(ea, 2),
        "rag_accuracy": round(ra, 2),
        "baseline_accuracy": round(ba, 2),
        "by_type": {t: dict(r) for t, r in type_results.items()},
    }
    out = _ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / "locomo_v2_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to results/locomo_v2_results.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--conversations", type=int, default=3)
    p.add_argument("--limit", type=int, default=150)
    args = p.parse_args()
    run_benchmark(args.conversations, args.limit)
