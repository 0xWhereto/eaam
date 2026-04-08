"""NRC VAD Lexicon mapping — categorical emotions to VAD vectors.

Based on the NRC Valence-Arousal-Dominance Lexicon v2 (Mohammad, 2018).
These are aggregate VAD centroids for each emotion category used by
the j-hartmann/emotion-english-distilroberta-base model.
"""

from eaam.models import VAD

# Mapping from emotion categories to VAD centroids.
# Sources: NRC VAD Lexicon v2, cross-referenced with Russell's circumplex model
# and Mehrabian's PAD research.
#
# valence:   [-1, 1]  negative to positive
# arousal:   [0, 1]   calm to excited
# dominance: [0, 1]   submissive to dominant

EMOTION_TO_VAD: dict[str, VAD] = {
    # j-hartmann model categories (7 emotions)
    "anger": VAD(valence=-0.80, arousal=0.80, dominance=0.55),
    "disgust": VAD(valence=-0.70, arousal=0.55, dominance=0.50),
    "fear": VAD(valence=-0.75, arousal=0.85, dominance=0.20),
    "joy": VAD(valence=0.90, arousal=0.65, dominance=0.70),
    "neutral": VAD(valence=0.0, arousal=0.15, dominance=0.50),
    "sadness": VAD(valence=-0.75, arousal=0.30, dominance=0.20),
    "surprise": VAD(valence=0.20, arousal=0.85, dominance=0.40),

    # Extended categories (GoEmotions 28-class, for future use)
    "admiration": VAD(valence=0.80, arousal=0.50, dominance=0.40),
    "amusement": VAD(valence=0.85, arousal=0.65, dominance=0.60),
    "annoyance": VAD(valence=-0.50, arousal=0.55, dominance=0.50),
    "approval": VAD(valence=0.65, arousal=0.35, dominance=0.60),
    "caring": VAD(valence=0.70, arousal=0.40, dominance=0.55),
    "confusion": VAD(valence=-0.20, arousal=0.50, dominance=0.25),
    "curiosity": VAD(valence=0.40, arousal=0.60, dominance=0.50),
    "desire": VAD(valence=0.60, arousal=0.70, dominance=0.50),
    "disappointment": VAD(valence=-0.60, arousal=0.35, dominance=0.25),
    "disapproval": VAD(valence=-0.55, arousal=0.45, dominance=0.55),
    "embarrassment": VAD(valence=-0.50, arousal=0.60, dominance=0.15),
    "excitement": VAD(valence=0.85, arousal=0.90, dominance=0.65),
    "gratitude": VAD(valence=0.85, arousal=0.45, dominance=0.45),
    "grief": VAD(valence=-0.85, arousal=0.40, dominance=0.10),
    "love": VAD(valence=0.95, arousal=0.65, dominance=0.50),
    "nervousness": VAD(valence=-0.40, arousal=0.75, dominance=0.20),
    "optimism": VAD(valence=0.75, arousal=0.55, dominance=0.65),
    "pride": VAD(valence=0.80, arousal=0.55, dominance=0.80),
    "realization": VAD(valence=0.30, arousal=0.60, dominance=0.55),
    "relief": VAD(valence=0.70, arousal=0.25, dominance=0.60),
    "remorse": VAD(valence=-0.65, arousal=0.35, dominance=0.15),
}


# Simple keyword-based emotion hints for the lexicon-only fallback.
# These are rough heuristics, not a full NLP pipeline.
LEXICON_KEYWORDS: dict[str, list[str]] = {
    "anger": ["angry", "furious", "rage", "hate", "frustrated", "annoyed", "mad", "pissed"],
    "disgust": ["disgusting", "gross", "repulsive", "revolting", "sick", "nasty", "awful"],
    "fear": ["afraid", "scared", "terrified", "anxious", "worried", "panic", "dread", "frightened"],
    "joy": ["happy", "glad", "delighted", "excited", "wonderful", "great", "amazing", "love", "excellent"],
    "sadness": ["sad", "depressed", "miserable", "unhappy", "grief", "loss", "lonely", "hopeless"],
    "surprise": ["surprised", "shocked", "unexpected", "wow", "astonished", "unbelievable", "suddenly"],
    "neutral": [],
}
