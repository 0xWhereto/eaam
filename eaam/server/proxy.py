"""Ollama/OpenAI-compatible proxy that adds emotion-anchored memory.

Sits between the user and the LLM, intercepting requests to:
1. Query memory for relevant context
2. Inject retrieved memories into the system prompt
3. Capture the conversation and encode new memories

Works with Ollama, LM Studio, llama.cpp server, or any OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import time
import uuid

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.consolidator import ConsolidationEngine
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.store.memory_store import MemoryStore

logger = logging.getLogger(__name__)

MEMORY_INJECTION_TEMPLATE = """## Associative Memory Context

The following memories were retrieved through emotion-anchored associative recall.
They are ranked by a composite of semantic similarity, emotional congruence with
the current conversation, and activation strength.

{memories}

---
Use these memories as context when they are relevant. Memories marked with higher
emotional scores may be especially pertinent to the current emotional tone of the conversation.
"""

MEMORY_ENTRY_TEMPLATE = """- [{score:.0%}] {content}
  (emotion: V={valence:.2f} A={arousal:.2f} D={dominance:.2f} | via: {path_info})"""


def create_proxy_app(config: EAAMConfig | None = None) -> FastAPI:
    """Create the FastAPI proxy application."""
    config = config or EAAMConfig.load()
    app = FastAPI(title="EAAM Proxy", version="0.1.0")

    # Initialize components
    store = MemoryStore(config)
    emotion_encoder = EmotionEncoder(config.emotion)
    encoding_pipeline = EncodingPipeline(store, emotion_encoder, config)
    retriever = AssociativeRetriever(store, emotion_encoder, config.retrieval)
    consolidator = ConsolidationEngine(store, config.consolidation)
    upstream = config.server.upstream_url
    conversation_ids: dict[str, str] = {}  # session tracking

    @app.post("/api/chat")
    async def ollama_chat(request: Request):
        """Ollama-native /api/chat endpoint with memory injection."""
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", True)

        # Extract user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return await _forward_ollama(upstream, body, stream)

        # Get or create conversation ID
        session_key = body.get("model", "default")
        if session_key not in conversation_ids:
            conversation_ids[session_key] = f"conv_{uuid.uuid4().hex[:8]}"
        conv_id = conversation_ids[session_key]

        # Encode user message into memory
        encoding_pipeline.encode(
            content=user_message,
            conversation_id=conv_id,
            role="user",
        )

        # Retrieve relevant memories
        results = retriever.retrieve(query=user_message, k=5)

        # Inject memories into system prompt
        if results:
            memory_text = _format_memories(results)
            body = _inject_memory_context(body, memory_text)

        # Forward to upstream
        if stream:
            return StreamingResponse(
                _stream_and_capture(upstream, body, encoding_pipeline, conv_id),
                media_type="application/x-ndjson",
            )
        else:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{upstream}/api/chat", json=body, timeout=120)
                resp_data = resp.json()

                # Capture assistant response
                assistant_msg = resp_data.get("message", {}).get("content", "")
                if assistant_msg:
                    encoding_pipeline.encode(
                        content=assistant_msg,
                        conversation_id=conv_id,
                        role="assistant",
                    )

                return resp_data

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        """OpenAI-compatible endpoint (works with LM Studio, llama.cpp, Ollama /v1/)."""
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Extract user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return await _forward_openai(upstream, body, stream)

        session_key = body.get("model", "default")
        if session_key not in conversation_ids:
            conversation_ids[session_key] = f"conv_{uuid.uuid4().hex[:8]}"
        conv_id = conversation_ids[session_key]

        # Encode + retrieve
        encoding_pipeline.encode(content=user_message, conversation_id=conv_id, role="user")
        results = retriever.retrieve(query=user_message, k=5)

        if results:
            memory_text = _format_memories(results)
            body = _inject_memory_context_openai(body, memory_text)

        # Forward
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{upstream}/v1/chat/completions", json=body, timeout=120)
            resp_data = resp.json()

            choices = resp_data.get("choices", [])
            if choices:
                assistant_msg = choices[0].get("message", {}).get("content", "")
                if assistant_msg:
                    encoding_pipeline.encode(content=assistant_msg, conversation_id=conv_id, role="assistant")

            return resp_data

    @app.get("/eaam/stats")
    async def get_stats():
        return store.stats()

    @app.get("/eaam/landscape")
    async def get_landscape():
        return consolidator.get_emotional_landscape()

    @app.post("/eaam/consolidate")
    async def trigger_consolidation():
        return consolidator.run()

    @app.post("/eaam/search")
    async def direct_search(request: Request):
        body = await request.json()
        results = retriever.retrieve(
            query=body["query"],
            k=body.get("k", 5),
            emotional_context=body.get("emotional_context"),
        )
        return [
            {
                "id": r.memory.id,
                "content": r.memory.content,
                "score": round(r.score, 4),
                "emotion": r.memory.emotion.to_dict(),
            }
            for r in results
        ]

    return app


def _format_memories(results) -> str:
    entries = []
    for r in results:
        path_info = "direct" if len(r.path) <= 1 else f"{len(r.path)-1}-hop chain"
        entries.append(MEMORY_ENTRY_TEMPLATE.format(
            score=r.score,
            content=r.memory.content[:200],
            valence=r.memory.emotion.valence,
            arousal=r.memory.emotion.arousal,
            dominance=r.memory.emotion.dominance,
            path_info=path_info,
        ))
    return MEMORY_INJECTION_TEMPLATE.format(memories="\n".join(entries))


def _inject_memory_context(body: dict, memory_text: str) -> dict:
    """Inject memory context into Ollama-format messages."""
    messages = body.get("messages", [])

    # Prepend to system message or create one
    has_system = any(m.get("role") == "system" for m in messages)
    if has_system:
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] = memory_text + "\n\n" + msg["content"]
                break
    else:
        messages.insert(0, {"role": "system", "content": memory_text})

    body["messages"] = messages
    return body


def _inject_memory_context_openai(body: dict, memory_text: str) -> dict:
    """Inject memory context into OpenAI-format messages."""
    return _inject_memory_context(body, memory_text)  # Same format


async def _forward_ollama(upstream: str, body: dict, stream: bool):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{upstream}/api/chat", json=body, timeout=120)
        return resp.json()


async def _forward_openai(upstream: str, body: dict, stream: bool):
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{upstream}/v1/chat/completions", json=body, timeout=120)
        return resp.json()


async def _stream_and_capture(upstream: str, body: dict, pipeline: EncodingPipeline, conv_id: str):
    """Stream response from upstream while capturing the full assistant message."""
    collected = []
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", f"{upstream}/api/chat", json=body, timeout=120) as resp:
            async for line in resp.aiter_lines():
                yield line + "\n"
                try:
                    chunk = json.loads(line)
                    if "message" in chunk and "content" in chunk["message"]:
                        collected.append(chunk["message"]["content"])
                except (json.JSONDecodeError, KeyError):
                    pass

    # Encode the full assistant response
    full_response = "".join(collected)
    if full_response.strip():
        pipeline.encode(content=full_response, conversation_id=conv_id, role="assistant")


def run_proxy(config: EAAMConfig | None = None):
    config = config or EAAMConfig.load()
    app = create_proxy_app(config)
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_level="info")
