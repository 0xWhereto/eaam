"""Run the xoanonxoloop-inspired loop engine on EAAM.

Usage: python3 tests/run_loop_engine.py [iterations]
"""

from __future__ import annotations

import sys
import tempfile

from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.loop_engine import run_loop_engine
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.store.memory_store import MemoryStore
from eaam.models import Edge, EdgeType

from tests.benchmark import MEMORY_CORPUS, QUERY_SCENARIOS, BaselineRAGRetriever
from tests.autotuner import score_benchmark, run_retrieval_with_params, TunableParams


# Pre-load the transformer model once
print("Loading transformer model...")
_config = EAAMConfig()
_config.emotion.use_transformer = True
SHARED_EMOTION = EmotionEncoder(_config.emotion)
print("Model loaded.\n")


def score_with_params(params_dict: dict) -> dict:
    """Build a fresh system with given params and score it."""
    config = EAAMConfig()
    config.emotion.use_transformer = True
    config.graph.persist_path = tempfile.mkdtemp()
    config.vector.persist_path = tempfile.mkdtemp()

    store = MemoryStore(config)
    pipeline = EncodingPipeline(store, SHARED_EMOTION, config)
    retriever = AssociativeRetriever(store, SHARED_EMOTION, config.retrieval)
    rag = BaselineRAGRetriever(store)

    # Convert flat params dict to TunableParams
    tp = TunableParams(**{k: v for k, v in params_dict.items() if hasattr(TunableParams, k)})

    # Patch encoding
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

    pipeline._build_semantic_edges = make_patched_semantic(tp, pipeline)
    pipeline._build_emotional_edges = make_patched_emotional(tp, pipeline)

    # Populate
    tag_index = {}
    for fixture in MEMORY_CORPUS:
        mem = pipeline.encode(content=fixture.content, conversation_id=fixture.conversation_id,
                              topic=fixture.topic, role=fixture.role)
        tag_index[mem.id] = set(fixture.tags)

    return score_benchmark(store, SHARED_EMOTION, pipeline, retriever, rag, tp, tag_index)


if __name__ == "__main__":
    n = 100
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n = int(sys.argv[1])

    # Load best params from previous auto-tuner as starting point
    initial = TunableParams()
    try:
        import yaml
        with open(_ROOT / "config" / "best_params.yaml") as f:
            saved = yaml.safe_load(f)
        if saved:
            initial = TunableParams(**{k: v for k, v in saved.items() if hasattr(TunableParams, k)})
            print(f"Loaded previous best params from config/best_params.yaml")
    except Exception:
        print("Using default params")

    initial_dict = initial.to_dict()

    best_params, best_score, ledger = run_loop_engine(
        score_fn=score_with_params,
        n_iterations=n,
        initial_params=initial_dict,
    )

    # Save
    import yaml
    with open(_ROOT / "config" / "loop_best_params.yaml", "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"\nBest params saved to config/loop_best_params.yaml")
