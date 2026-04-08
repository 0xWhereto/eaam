"""Emotion encoder — detects emotional content and produces VAD vectors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from eaam.emotion.vad_lexicon import EMOTION_TO_VAD, LEXICON_KEYWORDS
from eaam.models import VAD

if TYPE_CHECKING:
    from eaam.config import EmotionConfig

logger = logging.getLogger(__name__)


class EmotionEncoder:
    """Detects emotion in text and maps to VAD vectors.

    Two modes:
    - Transformer mode: uses HuggingFace model for emotion classification,
      then maps probabilities to VAD via weighted blend.
    - Lexicon mode: keyword matching fallback, no model needed.
    """

    def __init__(self, config: EmotionConfig | None = None):
        from eaam.config import EmotionConfig

        self.config = config or EmotionConfig()
        self._pipeline = None
        self._model_loaded = False

        if self.config.use_transformer:
            self._try_load_model()

    def _try_load_model(self):
        """Attempt to load the HuggingFace emotion model."""
        try:
            from transformers import pipeline

            logger.info("Loading emotion model: %s", self.config.model)
            self._pipeline = pipeline(
                "text-classification",
                model=self.config.model,
                top_k=None,  # return all class probabilities
                device=-1,  # CPU
            )
            self._model_loaded = True
            logger.info("Emotion model loaded successfully")
        except Exception as e:
            logger.warning("Failed to load emotion model, falling back to lexicon: %s", e)
            self._model_loaded = False

    def encode(self, text: str) -> VAD:
        """Analyze text and return a VAD emotional signature."""
        if self._model_loaded and self._pipeline is not None:
            return self._encode_transformer(text)
        return self._encode_lexicon(text)

    def encode_with_detail(self, text: str) -> tuple[VAD, dict[str, float]]:
        """Return both the VAD vector and the raw emotion probabilities."""
        if self._model_loaded and self._pipeline is not None:
            probs = self._get_emotion_probs(text)
            vad = self._probs_to_vad(probs)
            return vad, probs

        probs = self._lexicon_probs(text)
        vad = self._probs_to_vad(probs)
        return vad, probs

    def _encode_transformer(self, text: str) -> VAD:
        probs = self._get_emotion_probs(text)
        return self._probs_to_vad(probs)

    def _get_emotion_probs(self, text: str) -> dict[str, float]:
        """Get emotion probabilities from the transformer model."""
        # Truncate to avoid model max length issues
        truncated = text[:512]
        results = self._pipeline(truncated)
        # pipeline with top_k=None returns list of list of dicts
        if results and isinstance(results[0], list):
            results = results[0]
        return {r["label"]: r["score"] for r in results}

    def _encode_lexicon(self, text: str) -> VAD:
        probs = self._lexicon_probs(text)
        return self._probs_to_vad(probs)

    def _lexicon_probs(self, text: str) -> dict[str, float]:
        """Simple keyword-based emotion scoring."""
        text_lower = text.lower()
        scores: dict[str, float] = {}

        for emotion, keywords in LEXICON_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[emotion] = float(count)

        total = sum(scores.values())
        if total == 0:
            return {"neutral": 1.0}

        return {k: v / total for k, v in scores.items() if v > 0}

    @staticmethod
    def _probs_to_vad(probs: dict[str, float]) -> VAD:
        """Convert emotion probability distribution to VAD vector via weighted blend."""
        v, a, d = 0.0, 0.0, 0.0
        total_weight = 0.0

        for emotion, prob in probs.items():
            if emotion in EMOTION_TO_VAD and prob > 0.01:
                vad = EMOTION_TO_VAD[emotion]
                v += vad.valence * prob
                a += vad.arousal * prob
                d += vad.dominance * prob
                total_weight += prob

        if total_weight == 0:
            return VAD.neutral()

        return VAD(
            valence=max(-1.0, min(1.0, v / total_weight)),
            arousal=max(0.0, min(1.0, a / total_weight)),
            dominance=max(0.0, min(1.0, d / total_weight)),
        )
