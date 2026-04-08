"""Core data models for EAAM."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class VAD:
    """Valence-Arousal-Dominance emotional signature.

    valence:   [-1, 1]  negative to positive
    arousal:   [0, 1]   calm to excited
    dominance: [0, 1]   submissive to dominant
    """

    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0

    def __post_init__(self):
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(0.0, min(1.0, self.dominance))

    def similarity(self, other: VAD) -> float:
        """Normalized Euclidean similarity in VAD space, returns [0, 1]."""
        dist = math.sqrt(
            (self.valence - other.valence) ** 2
            + (self.arousal - other.arousal) ** 2
            + (self.dominance - other.dominance) ** 2
        )
        max_dist = math.sqrt(4 + 1 + 1)  # valence range=2, arousal/dominance range=1
        return 1.0 - (dist / max_dist)

    def to_dict(self) -> dict:
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}

    @classmethod
    def from_dict(cls, d: dict) -> VAD:
        return cls(valence=d["valence"], arousal=d["arousal"], dominance=d["dominance"])

    @classmethod
    def neutral(cls) -> VAD:
        return cls(valence=0.0, arousal=0.2, dominance=0.5)


class EdgeType(str, Enum):
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    THEMATIC = "thematic"
    REFLECTION = "reflection"  # link from reflection node to source memories


@dataclass
class Edge:
    """Weighted association between two memories."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float  # [0, 1]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            edge_type=EdgeType(d["edge_type"]),
            weight=d["weight"],
            metadata=d.get("metadata", {}),
        )


@dataclass
class Memory:
    """A single memory node with emotional signature and activation dynamics."""

    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:10]}")
    content: str = ""
    summary: str = ""

    # Emotional signature
    emotion: VAD = field(default_factory=VAD.neutral)

    # Activation dynamics
    base_activation: float = 0.5
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Context
    conversation_id: str = ""
    topic: str = ""
    role: str = ""  # "user" or "assistant"
    is_reflection: bool = False  # True for consolidated reflection nodes

    def effective_activation(self, decay_rate: float = 0.95, half_life_hours: float = 72.0) -> float:
        """Compute current activation level with time decay.

        Mirrors human memory: activation = base * decay^(hours_since_access)
        High-arousal memories decay slower (arousal acts as consolidation strength).
        """
        hours_elapsed = (time.time() - self.last_accessed) / 3600.0
        arousal_factor = 1.0 + self.emotion.arousal  # [1.0, 2.0]
        adjusted_half_life = half_life_hours * arousal_factor
        decay = decay_rate ** (hours_elapsed / adjusted_half_life)
        frequency_boost = math.log1p(self.access_count) * 0.1
        return min(self.base_activation * decay + frequency_boost, 1.0)

    def touch(self):
        """Mark this memory as accessed (retrieval strengthens it)."""
        self.access_count += 1
        self.last_accessed = time.time()
        # Retrieval strengthens base activation (testing effect)
        self.base_activation = min(self.base_activation + 0.02, 1.0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "emotion": self.emotion.to_dict(),
            "base_activation": self.base_activation,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "conversation_id": self.conversation_id,
            "topic": self.topic,
            "role": self.role,
            "is_reflection": self.is_reflection,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Memory:
        return cls(
            id=d["id"],
            content=d["content"],
            summary=d.get("summary", ""),
            emotion=VAD.from_dict(d["emotion"]),
            base_activation=d["base_activation"],
            access_count=d["access_count"],
            created_at=d["created_at"],
            last_accessed=d["last_accessed"],
            conversation_id=d.get("conversation_id", ""),
            topic=d.get("topic", ""),
            role=d.get("role", ""),
            is_reflection=d.get("is_reflection", False),
        )


@dataclass
class RetrievalResult:
    """A memory returned from retrieval with scoring breakdown."""

    memory: Memory
    score: float
    semantic_score: float = 0.0
    emotional_score: float = 0.0
    activation_score: float = 0.0
    spreading_score: float = 0.0
    path: list[str] = field(default_factory=list)  # association chain that led here
