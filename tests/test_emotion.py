"""Tests for emotion encoder."""

from eaam.config import EmotionConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.models import VAD


def test_lexicon_fallback_detects_joy():
    config = EmotionConfig(use_transformer=False)
    encoder = EmotionEncoder(config)
    vad = encoder.encode("I'm so happy and excited about this wonderful day!")
    assert vad.valence > 0.5
    assert vad.arousal > 0.4


def test_lexicon_fallback_detects_anger():
    config = EmotionConfig(use_transformer=False)
    encoder = EmotionEncoder(config)
    vad = encoder.encode("I'm absolutely furious and frustrated about this")
    assert vad.valence < -0.3
    assert vad.arousal > 0.5


def test_lexicon_fallback_neutral():
    config = EmotionConfig(use_transformer=False)
    encoder = EmotionEncoder(config)
    vad = encoder.encode("The meeting is at 3pm in the conference room")
    assert -0.3 <= vad.valence <= 0.3


def test_encode_with_detail():
    config = EmotionConfig(use_transformer=False)
    encoder = EmotionEncoder(config)
    vad, probs = encoder.encode_with_detail("I'm scared and anxious about the deadline")
    assert isinstance(vad, VAD)
    assert isinstance(probs, dict)
    assert "fear" in probs


def test_probs_to_vad_empty():
    result = EmotionEncoder._probs_to_vad({})
    assert result.valence == 0.0
    assert result.arousal == 0.2
    assert result.dominance == 0.5
