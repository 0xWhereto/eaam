"""Encoding pipeline — processes new interactions into emotionally-tagged memories."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Callable

from eaam.emotion.encoder import EmotionEncoder
from eaam.models import Edge, EdgeType, Memory, VAD
from eaam.store.memory_store import MemoryStore

if TYPE_CHECKING:
    from eaam.config import EAAMConfig

logger = logging.getLogger(__name__)

# Time window for temporal associations (seconds)
TEMPORAL_WINDOW = 600  # 10 minutes

Redactor = Callable[[str], str]

_AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_API_KEY_RE = re.compile(r"\b(?:sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,})\b")
_GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _is_luhn_valid(number: str) -> bool:
    """Return whether a digit string satisfies the Luhn checksum."""
    if len(number) < 13 or len(number) > 19:
        return False

    total = 0
    parity = len(number) % 2
    for index, char in enumerate(number):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _redact_credit_cards(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if _is_luhn_valid(digits):
            return "[REDACTED_CREDIT_CARD]"
        return match.group(0)

    return _CARD_CANDIDATE_RE.sub(replace, text)


def redact_sensitive_content(content: str) -> str:
    """Redact common secrets and PII before content enters long-term memory."""
    redacted = _redact_credit_cards(content)
    replacements = (
        (_AWS_KEY_RE, "[REDACTED_AWS_KEY]"),
        (_API_KEY_RE, "[REDACTED_API_KEY]"),
        (_GITHUB_TOKEN_RE, "[REDACTED_GITHUB_TOKEN]"),
        (_JWT_RE, "[REDACTED_JWT]"),
        (_SSN_RE, "[REDACTED_SSN]"),
    )

    for pattern, marker in replacements:
        redacted = pattern.sub(marker, redacted)

    return redacted


class EncodingPipeline:
    """Encodes new text into the memory system with emotional tagging and association building."""

    def __init__(
        self,
        store: MemoryStore,
        emotion_encoder: EmotionEncoder,
        config: EAAMConfig | None = None,
        redactor: Redactor | None = redact_sensitive_content,
    ):
        from eaam.config import EAAMConfig

        self.store = store
        self.emotion = emotion_encoder
        self.config = config or EAAMConfig()
        self.redactor = redactor

    def encode(
        self,
        content: str,
        conversation_id: str = "",
        role: str = "user",
        topic: str = "",
        override_emotion: VAD | None = None,
    ) -> Memory:
        """Encode a piece of text into the memory system.

        1. Detect emotion -> VAD vector
        2. Compute initial base activation (arousal-modulated)
        3. Create memory node
        4. Store in graph + vector
        5. Build associations to existing memories
        6. Persist
        """
        # Validate input
        if not content or not content.strip():
            raise ValueError("Cannot encode empty content")

        if self.redactor is not None:
            redacted_content = self.redactor(content)
            if redacted_content != content:
                logger.info("Sensitive content redacted before memory encoding")
            content = redacted_content

        if not content or not content.strip():
            raise ValueError("Cannot encode empty content after redaction")

        # Step 1: Emotion detection
        if override_emotion is not None:
            vad = override_emotion
            emotion_detail = {"override": True}
        else:
            vad, emotion_detail = self.emotion.encode_with_detail(content)

        logger.info(
            "Emotion detected: V=%.2f A=%.2f D=%.2f (top: %s)",
            vad.valence,
            vad.arousal,
            vad.dominance,
            max(emotion_detail, key=emotion_detail.get) if emotion_detail else "none",
        )

        # Step 2: Arousal-modulated base activation
        # High arousal = stronger initial encoding (mirrors norepinephrine effect)
        base_activation = 0.4 + (0.5 * vad.arousal)

        # Step 3: Create memory
        memory = Memory(
            content=content,
            summary=content[:100] + ("..." if len(content) > 100 else ""),
            emotion=vad,
            base_activation=base_activation,
            conversation_id=conversation_id,
            topic=topic,
            role=role,
        )

        # Step 4: Store
        self.store.add(memory)

        # Step 5: Build associations
        self._build_associations(memory)

        # Step 6: Persist
        self.store.save()

        logger.info("Memory encoded: %s (activation=%.2f)", memory.id, base_activation)
        return memory

    def _build_associations(self, new_memory: Memory):
        """Build multi-type associations between the new memory and existing ones."""
        self._build_semantic_edges(new_memory)
        self._build_emotional_edges(new_memory)
        self._build_temporal_edges(new_memory)

    def _build_semantic_edges(self, memory: Memory):
        """Connect to semantically similar memories."""
        candidates = self.store.semantic_search(memory.content, n=10)

        for mem_id, similarity in candidates:
            if mem_id == memory.id:
                continue
            if similarity < 0.3:  # threshold for meaningful semantic similarity
                continue

            self.store.add_edge(Edge(
                source_id=memory.id,
                target_id=mem_id,
                edge_type=EdgeType.SEMANTIC,
                weight=similarity,
            ))

    def _build_emotional_edges(self, memory: Memory):
        """Connect to memories with similar emotional signatures.

        This is the key differentiator — memories about completely different
        topics get linked if they share emotional resonance.
        """
        emotional_neighbors = self.store.emotional_search(
            memory.emotion, threshold=0.75, limit=8
        )

        # Hoist semantic search outside the loop (was BUG-4: repeated N times)
        semantic_candidates = dict(self.store.semantic_search(memory.content, n=20))

        for neighbor, emo_sim in emotional_neighbors:
            if neighbor.id == memory.id:
                continue

            # Only create emotional edges where semantic similarity is LOW
            # (otherwise the semantic edge already covers the relationship).
            # This specifically captures cross-domain emotional associations.
            semantic_sim = semantic_candidates.get(neighbor.id, 0.0)

            if semantic_sim < 0.5:  # different topic, same emotion
                self.store.add_edge(Edge(
                    source_id=memory.id,
                    target_id=neighbor.id,
                    edge_type=EdgeType.EMOTIONAL,
                    weight=emo_sim,
                    metadata={
                        "shared_valence": (memory.emotion.valence + neighbor.emotion.valence) / 2,
                        "semantic_distance": 1.0 - semantic_sim,
                    },
                ))

    def _build_temporal_edges(self, memory: Memory):
        """Connect to memories created in the same time window or conversation."""
        # Same conversation — strong temporal link
        if memory.conversation_id:
            conv_memories = self.store.graph.find_by_conversation(memory.conversation_id)
            for other in conv_memories:
                if other.id == memory.id:
                    continue
                time_gap = abs(memory.created_at - other.created_at)
                # Weight decays with time gap, but conversation membership guarantees a base
                weight = max(0.3, 1.0 - (time_gap / TEMPORAL_WINDOW))
                self.store.add_edge(Edge(
                    source_id=memory.id,
                    target_id=other.id,
                    edge_type=EdgeType.TEMPORAL,
                    weight=weight,
                    metadata={"gap_seconds": time_gap},
                ))
            return

        # No conversation ID — look for recent memories within time window
        now = memory.created_at
        for other in self.store.graph.get_all_memories():
            if other.id == memory.id:
                continue
            time_gap = abs(now - other.created_at)
            if time_gap <= TEMPORAL_WINDOW:
                weight = 1.0 - (time_gap / TEMPORAL_WINDOW)
                self.store.add_edge(Edge(
                    source_id=memory.id,
                    target_id=other.id,
                    edge_type=EdgeType.TEMPORAL,
                    weight=max(0.1, weight),
                    metadata={"gap_seconds": time_gap},
                ))
