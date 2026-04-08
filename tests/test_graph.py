"""Tests for in-memory graph store."""

import tempfile

from eaam.config import GraphConfig
from eaam.models import Edge, EdgeType, Memory, VAD
from eaam.store.graph import GraphStore


def _make_store() -> GraphStore:
    tmpdir = tempfile.mkdtemp()
    return GraphStore(GraphConfig(persist_path=tmpdir))


def test_add_and_get():
    store = _make_store()
    mem = Memory(id="test1", content="hello")
    store.add_memory(mem)
    assert store.get_memory("test1") is not None
    assert store.get_memory("test1").content == "hello"


def test_add_edge_and_neighbors():
    store = _make_store()
    store.add_memory(Memory(id="a", content="alpha"))
    store.add_memory(Memory(id="b", content="beta"))

    store.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC, weight=0.9))

    neighbors = store.get_neighbors("a")
    assert len(neighbors) == 1
    assert neighbors[0][0].id == "b"
    assert neighbors[0][1].weight == 0.9


def test_bidirectional_traversal():
    store = _make_store()
    store.add_memory(Memory(id="a", content="alpha"))
    store.add_memory(Memory(id="b", content="beta"))

    store.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.EMOTIONAL, weight=0.8))

    # Should find b from a (outgoing)
    assert len(store.get_neighbors("a")) == 1
    # Should also find a from b (incoming, treated as undirected)
    assert len(store.get_neighbors("b")) == 1


def test_emotional_search():
    store = _make_store()
    store.add_memory(Memory(id="happy1", content="great day", emotion=VAD(valence=0.8, arousal=0.6, dominance=0.7)))
    store.add_memory(Memory(id="sad1", content="terrible day", emotion=VAD(valence=-0.8, arousal=0.3, dominance=0.2)))
    store.add_memory(Memory(id="happy2", content="wonderful", emotion=VAD(valence=0.7, arousal=0.5, dominance=0.6)))

    results = store.find_by_emotion(VAD(valence=0.75, arousal=0.55, dominance=0.65), threshold=0.8)
    ids = [m.id for m, _ in results]
    assert "happy1" in ids
    assert "happy2" in ids
    assert "sad1" not in ids


def test_persistence():
    tmpdir = tempfile.mkdtemp()
    config = GraphConfig(persist_path=tmpdir)

    store1 = GraphStore(config)
    store1.add_memory(Memory(id="persist_test", content="survives restart"))
    store1.add_edge(Edge(source_id="persist_test", target_id="persist_test", edge_type=EdgeType.SEMANTIC, weight=0.5))
    store1.save()

    store2 = GraphStore(config)
    assert store2.get_memory("persist_test") is not None
    assert store2.count() == 1


def test_prune_edges():
    store = _make_store()
    store.add_memory(Memory(id="a", content="a"))
    store.add_memory(Memory(id="b", content="b"))

    store.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.SEMANTIC, weight=0.1))
    store.add_edge(Edge(source_id="a", target_id="b", edge_type=EdgeType.EMOTIONAL, weight=0.9))

    store.delete_edges_below(0.5)
    assert len(store.get_all_edges()) == 1
    assert store.get_all_edges()[0].weight == 0.9
