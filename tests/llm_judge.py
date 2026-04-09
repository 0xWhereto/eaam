"""LLM Judge — shared answer evaluation module for EAAM benchmarks.

Replaces heuristic word-overlap scoring with actual LLM reasoning.
Supports multiple providers via simple HTTP calls (no SDK dependencies).

Usage:
    from llm_judge import LLMJudge

    judge = LLMJudge()  # auto-detects available provider
    idx = judge.pick_answer(question, choices, retrieved_texts)
    score = judge.score_free_response(question, expected, actual)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

PICK_ANSWER_PROMPT = """\
You are evaluating a multiple-choice question using retrieved memory context.

## Retrieved Context
{context}

## Question
{question}

## Choices
{choices}

Based ONLY on the retrieved context above, which choice best answers the question?
If the context doesn't contain enough information, make your best guess.

Respond with ONLY the choice number (0-indexed). Nothing else."""

SCORE_RESPONSE_PROMPT = """\
You are evaluating whether a response correctly answers a question about past conversations.

## Question
{question}

## Expected Answer
{expected}

## Actual Response
{actual}

Score the response on a scale of 0-100:
- 100: Perfectly captures the expected answer
- 75: Mostly correct with minor details missing
- 50: Partially correct
- 25: Tangentially related but mostly wrong
- 0: Completely wrong or irrelevant

Respond with ONLY a JSON object: {{"score": <number>, "reason": "<brief explanation>"}}"""

JUDGE_RELEVANCE_PROMPT = """\
You are evaluating whether retrieved memories are relevant to answering a question.

## Question
{question}

## Retrieved Memories
{memories}

For each memory (numbered), rate its relevance to answering the question.

Respond with ONLY a JSON object:
{{"relevance": [<score_0>, <score_1>, ...], "best_memory_idx": <index>}}
Where each score is 0-100 (0=irrelevant, 100=directly answers the question)."""


@dataclass
class JudgeResult:
    answer_idx: int
    confidence: float
    raw_response: str
    latency_ms: float


@dataclass
class ScoreResult:
    score: float
    reason: str
    latency_ms: float


class LLMJudge:
    """LLM-based answer evaluation for benchmarks."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 5,
    ):
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=60.0)

        resolved = self._resolve_provider(provider, model, api_key, base_url)
        self.provider = resolved["provider"]
        self.model = resolved["model"]
        self.api_key = resolved["api_key"]
        self.base_url = resolved["base_url"]

        logger.info(f"LLM Judge initialized: {self.provider}/{self.model}")

    def _resolve_provider(
        self,
        provider: Optional[str],
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> dict:
        if provider and model and api_key:
            return {
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "base_url": base_url or self._default_base_url(provider),
            }

        if os.environ.get("ANTHROPIC_API_KEY"):
            return {
                "provider": "anthropic",
                "model": model or "claude-sonnet-4-20250514",
                "api_key": os.environ["ANTHROPIC_API_KEY"],
                "base_url": base_url or "https://api.anthropic.com",
            }
        if os.environ.get("OPENAI_API_KEY"):
            return {
                "provider": "openai",
                "model": model or "gpt-4o-mini",
                "api_key": os.environ["OPENAI_API_KEY"],
                "base_url": base_url or "https://api.openai.com/v1",
            }

        # Ollama fallback
        try:
            r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
            if r.status_code == 200:
                models = r.json().get("models", [])
                if models:
                    return {
                        "provider": "ollama",
                        "model": model or models[0]["name"],
                        "api_key": "",
                        "base_url": "http://localhost:11434",
                    }
        except Exception:
            pass

        raise RuntimeError(
            "No LLM provider found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "or run Ollama locally."
        )

    def _default_base_url(self, provider: str) -> str:
        return {
            "anthropic": "https://api.anthropic.com",
            "openai": "https://api.openai.com/v1",
            "ollama": "http://localhost:11434",
        }.get(provider, "http://localhost:11434")

    def _call_llm(self, prompt: str) -> str:
        max_attempts = max(self.max_retries + 1, 10)
        for attempt in range(max_attempts):
            try:
                if self.provider == "anthropic":
                    return self._call_anthropic(prompt)
                elif self.provider == "ollama":
                    return self._call_ollama(prompt)
                else:
                    return self._call_openai_compat(prompt)
            except Exception as e:
                err = str(e)
                is_retryable = any(code in err for code in ("429", "529", "500", "502", "503"))
                if attempt == max_attempts - 1 or not is_retryable:
                    raise
                wait = min(2 ** attempt * 3.0 + 5.0, 90.0)
                print(f"  [judge] retry {attempt+1}/{max_attempts} in {wait:.0f}s ({err[:80]})")
                time.sleep(wait)
        return ""

    def _call_anthropic(self, prompt: str) -> str:
        r = self._client.post(
            f"{self.base_url}/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 256,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    def _call_openai_compat(self, prompt: str) -> str:
        r = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, prompt: str) -> str:
        r = self._client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
        )
        r.raise_for_status()
        return r.json()["response"]

    def pick_answer(
        self,
        question: str,
        choices: list[str],
        retrieved_texts: list[str],
    ) -> JudgeResult:
        trimmed = [t[:300] for t in retrieved_texts[:8]]
        context = "\n".join(
            f"[Memory {i+1}] {t}" for i, t in enumerate(trimmed)
        )
        choices_str = "\n".join(f"{i}: {c}" for i, c in enumerate(choices))

        prompt = PICK_ANSWER_PROMPT.format(
            context=context, question=question, choices=choices_str
        )

        t0 = time.time()
        try:
            raw = self._call_llm(prompt)
        except Exception as e:
            print(f"  [judge] FAILED, falling back to random: {str(e)[:60]}")
            import random
            return JudgeResult(
                answer_idx=random.randint(0, len(choices) - 1),
                confidence=0.0,
                raw_response=f"ERROR: {e}",
                latency_ms=(time.time() - t0) * 1000,
            )
        latency = (time.time() - t0) * 1000

        idx = self._parse_int(raw, max_val=len(choices) - 1)
        return JudgeResult(
            answer_idx=idx,
            confidence=1.0,
            raw_response=raw.strip(),
            latency_ms=latency,
        )

    def score_free_response(
        self,
        question: str,
        expected: str,
        actual: str,
    ) -> ScoreResult:
        prompt = SCORE_RESPONSE_PROMPT.format(
            question=question, expected=expected, actual=actual[:500]
        )

        t0 = time.time()
        try:
            raw = self._call_llm(prompt)
        except Exception as e:
            print(f"  [judge] score FAILED: {str(e)[:60]}")
            return ScoreResult(score=0.0, reason=f"ERROR: {e}", latency_ms=(time.time() - t0) * 1000)
        latency = (time.time() - t0) * 1000

        score, reason = self._parse_score_json(raw)
        return ScoreResult(score=score, reason=reason, latency_ms=latency)

    def rate_relevance(
        self,
        question: str,
        memories: list[str],
    ) -> list[float]:
        mem_str = "\n".join(f"[{i}] {m}" for i, m in enumerate(memories))
        prompt = JUDGE_RELEVANCE_PROMPT.format(question=question, memories=mem_str)

        raw = self._call_llm(prompt)
        try:
            data = json.loads(self._extract_json(raw))
            return [float(s) / 100.0 for s in data.get("relevance", [])]
        except Exception:
            return [0.5] * len(memories)

    def _parse_int(self, text: str, max_val: int) -> int:
        nums = re.findall(r"\d+", text.strip())
        if nums:
            val = int(nums[0])
            return min(val, max_val)
        return 0

    def _parse_score_json(self, text: str) -> tuple[float, str]:
        try:
            data = json.loads(self._extract_json(text))
            return float(data.get("score", 0)), data.get("reason", "")
        except Exception:
            nums = re.findall(r"\d+", text)
            return float(nums[0]) if nums else 0.0, text.strip()

    def _extract_json(self, text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text
