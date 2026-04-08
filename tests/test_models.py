"""Tests for core data models."""

from eaam.models import VAD, Edge, EdgeType, Memory


def test_vad_similarity_identical():
    a = VAD(valence=0.5, arousal=0.5, dominance=0.5)
    b = VAD(valence=0.5, arousal=0.5, dominance=0.5)
    assert a.similarity(b) == 1.0


def test_vad_similarity_opposite():
    a = VAD(valence=1.0, arousal=1.0, dominance=1.0)
    b = VAD(valence=-1.0, arousal=0.0, dominance=0.0)
    sim = a.similarity(b)
    assert sim < 0.3  # very different


def test_vad_serialization():
    vad = VAD(valence=-0.5, arousal=0.8, dominance=0.3)
    d = vad.to_dict()
    restored = VAD.from_dict(d)
    assert restored.valence == vad.valence
    assert restored.arousal == vad.arousal
    assert restored.dominance == vad.dominance


def test_memory_activation_decays():
    import time

    mem = Memory(content="test", emotion=VAD(arousal=0.5))
    mem.last_accessed = time.time() - 3600 * 24  # 24 hours ago
    assert mem.effective_activation() < mem.base_activation


def test_memory_touch_strengthens():
    mem = Memory(content="test")
    initial = mem.base_activation
    mem.touch()
    assert mem.base_activation > initial
    assert mem.access_count == 1


def test_memory_serialization():
    mem = Memory(content="hello world", topic="test", role="user")
    d = mem.to_dict()
    restored = Memory.from_dict(d)
    assert restored.content == mem.content
    assert restored.topic == mem.topic
    assert restored.id == mem.id


def test_edge_serialization():
    edge = Edge(source_id="a", target_id="b", edge_type=EdgeType.EMOTIONAL, weight=0.85)
    d = edge.to_dict()
    restored = Edge.from_dict(d)
    assert restored.source_id == "a"
    assert restored.edge_type == EdgeType.EMOTIONAL
    assert restored.weight == 0.85
