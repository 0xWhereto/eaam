"""Configuration for EAAM."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class GraphConfig:
    backend: str = "memory"  # "memory" (JSON-persisted), "falkordb", "neo4j"
    persist_path: str = ""

    def __post_init__(self):
        if not self.persist_path:
            self.persist_path = os.path.join(_default_data_dir(), "graph")


@dataclass
class EmotionConfig:
    model: str = "j-hartmann/emotion-english-distilroberta-base"
    use_transformer: bool = True  # False = lexicon-only fallback (no GPU needed)


@dataclass
class EmbeddingConfig:
    model: str = "all-MiniLM-L6-v2"  # sentence-transformers default, small and fast
    dimensions: int = 384


@dataclass
class RetrievalConfig:
    alpha: float = 0.35   # semantic similarity weight
    beta: float = 0.30    # emotional congruence weight
    gamma: float = 0.20   # base activation weight
    delta: float = 0.15   # spreading activation weight
    activation_hops: int = 2
    activation_decay: float = 0.6
    candidate_pool: int = 20  # vector search candidates before spreading


@dataclass
class ConsolidationConfig:
    interval_hours: float = 6.0
    decay_rate: float = 0.95
    strengthen_threshold: float = 0.7  # arousal above this gets strengthened
    prune_edge_threshold: float = 0.15
    min_cluster_size: int = 3


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8800
    upstream_url: str = "http://localhost:11434"  # Ollama default


@dataclass
class VectorConfig:
    backend: str = "chroma"
    persist_path: str = ""

    def __post_init__(self):
        if not self.persist_path:
            self.persist_path = os.path.join(_default_data_dir(), "vectors")


@dataclass
class EAAMConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> EAAMConfig:
        if path is None:
            path = Path(_default_config_path())
        else:
            path = Path(path)

        if not path.exists():
            return cls()

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        config = cls()
        if "graph" in raw:
            config.graph = GraphConfig(**raw["graph"])
        if "emotion" in raw:
            config.emotion = EmotionConfig(**raw["emotion"])
        if "embedding" in raw:
            config.embedding = EmbeddingConfig(**raw["embedding"])
        if "retrieval" in raw:
            config.retrieval = RetrievalConfig(**raw["retrieval"])
        if "consolidation" in raw:
            config.consolidation = ConsolidationConfig(**raw["consolidation"])
        if "server" in raw:
            config.server = ServerConfig(**raw["server"])
        if "vector" in raw:
            config.vector = VectorConfig(**raw["vector"])
        return config

    def save(self, path: str | Path | None = None):
        if path is None:
            path = Path(_default_config_path())
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "graph": {"backend": self.graph.backend, "persist_path": self.graph.persist_path},
            "emotion": {"model": self.emotion.model, "use_transformer": self.emotion.use_transformer},
            "embedding": {"model": self.embedding.model, "dimensions": self.embedding.dimensions},
            "retrieval": {
                "alpha": self.retrieval.alpha,
                "beta": self.retrieval.beta,
                "gamma": self.retrieval.gamma,
                "delta": self.retrieval.delta,
                "activation_hops": self.retrieval.activation_hops,
                "activation_decay": self.retrieval.activation_decay,
                "candidate_pool": self.retrieval.candidate_pool,
            },
            "consolidation": {
                "interval_hours": self.consolidation.interval_hours,
                "decay_rate": self.consolidation.decay_rate,
                "prune_edge_threshold": self.consolidation.prune_edge_threshold,
                "min_cluster_size": self.consolidation.min_cluster_size,
            },
            "server": {
                "host": self.server.host,
                "port": self.server.port,
                "upstream_url": self.server.upstream_url,
            },
            "vector": {"backend": self.vector.backend, "persist_path": self.vector.persist_path},
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _default_data_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".eaam", "data")


def _default_config_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".eaam", "config.yaml")
