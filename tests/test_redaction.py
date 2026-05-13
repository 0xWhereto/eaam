"""Tests for pre-encode sensitive content redaction."""

from eaam.engine.encoder import EncodingPipeline, redact_sensitive_content
from eaam.models import VAD


class _FakeGraph:
    def find_by_conversation(self, conversation_id):
        return []

    def get_all_memories(self):
        return []


class _FakeStore:
    def __init__(self):
        self.graph = _FakeGraph()
        self.added = []
        self.saved = False
        self.semantic_queries = []

    def add(self, memory):
        self.added.append(memory)

    def semantic_search(self, query, n=20):
        self.semantic_queries.append(query)
        return []

    def emotional_search(self, target_vad, threshold=0.75, limit=10):
        return []

    def add_edge(self, edge):
        raise AssertionError("No edges should be added in this test")

    def save(self):
        self.saved = True


class _FakeEmotionEncoder:
    def __init__(self):
        self.seen_text = None

    def encode_with_detail(self, text):
        self.seen_text = text
        return VAD.neutral(), {"neutral": 1.0}


def test_default_redactor_masks_common_secret_patterns():
    raw = (
        "AWS AKIA1234567890ABCDEF, GitHub ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ, "
        "JWT eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signature, "
        "SSN 123-45-6789, card 4242 4242 4242 4242, key sk-ant-api03-exampleSecretTokenValue"
    )

    redacted = redact_sensitive_content(raw)

    assert "AKIA1234567890ABCDEF" not in redacted
    assert "ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ" not in redacted
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert "123-45-6789" not in redacted
    assert "4242 4242 4242 4242" not in redacted
    assert "sk-ant-api03-exampleSecretTokenValue" not in redacted
    assert "[REDACTED_AWS_KEY]" in redacted
    assert "[REDACTED_GITHUB_TOKEN]" in redacted
    assert "[REDACTED_JWT]" in redacted
    assert "[REDACTED_SSN]" in redacted
    assert "[REDACTED_CREDIT_CARD]" in redacted
    assert "[REDACTED_API_KEY]" in redacted


def test_default_redactor_keeps_non_luhn_digit_sequences():
    redacted = redact_sensitive_content("tracking id 1234 5678 9012 3456 should remain")

    assert "1234 5678 9012 3456" in redacted
    assert "[REDACTED_CREDIT_CARD]" not in redacted


def test_encoding_pipeline_redacts_before_every_storage_sink():
    store = _FakeStore()
    emotion = _FakeEmotionEncoder()
    pipeline = EncodingPipeline(store, emotion)

    memory = pipeline.encode("remember card 4242 4242 4242 4242")

    assert "4242 4242 4242 4242" not in memory.content
    assert memory.content == "remember card [REDACTED_CREDIT_CARD]"
    assert store.added[0].content == memory.content
    assert store.semantic_queries == [memory.content, memory.content]
    assert emotion.seen_text == memory.content
    assert store.saved is True


def test_encoding_pipeline_accepts_custom_redactor():
    store = _FakeStore()
    emotion = _FakeEmotionEncoder()
    pipeline = EncodingPipeline(store, emotion, redactor=lambda text: text.replace("private", "public"))

    memory = pipeline.encode("private note")

    assert memory.content == "public note"
    assert emotion.seen_text == "public note"


def test_encoding_pipeline_can_opt_out_of_redaction():
    store = _FakeStore()
    emotion = _FakeEmotionEncoder()
    pipeline = EncodingPipeline(store, emotion, redactor=None)

    memory = pipeline.encode("raw key sk-testSecretValueThatWouldNormallyRedact")

    assert memory.content == "raw key sk-testSecretValueThatWouldNormallyRedact"
    assert emotion.seen_text == memory.content
