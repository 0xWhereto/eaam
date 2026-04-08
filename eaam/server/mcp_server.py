"""MCP Server for EAAM — emotion-anchored associative memory for any AI.

Exposes memory tools that let the host AI (Claude, Cursor, etc.) store
and retrieve memories with automatic emotional tagging. The AI becomes
the reasoning layer; EAAM handles storage, emotion detection, association
building, and multi-pathway retrieval.

Run with: eaam serve --mode mcp
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.consolidator import ConsolidationEngine
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.models import VAD
from eaam.store.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# System prompt injected as a resource so the host AI knows how to use EAAM
EAAM_SYSTEM_PROMPT = """You have access to EAAM (Emotion-Anchored Associative Memory) — a long-term memory system that detects emotions and builds associations between memories.

HOW TO USE MEMORY:

1. STORE memories when the user shares something meaningful:
   - Personal experiences, frustrations, wins, fears, goals
   - Technical problems and their solutions
   - Important decisions and their context
   - Anything with emotional significance
   Call memory_store with the key content. The system auto-detects emotions.

2. SEARCH memories when context from past conversations would help:
   - Before answering, check if relevant past context exists
   - Include the user's current emotional tone in emotional_context
   - The system retrieves across 4 pathways: semantic, emotional, spreading activation, and involuntary
   - Memories matching the current MOOD surface even if topics differ
   Call memory_search with the query and emotional_context.

3. BE PROACTIVE:
   - Store important moments WITHOUT being asked
   - Search memory BEFORE answering when the user references past events
   - If the user seems frustrated, search for past frustration moments — the context may help
   - If the user is celebrating, recall past wins to share in the moment

4. EMOTIONAL AWARENESS:
   - The system tags every memory with Valence (positive/negative), Arousal (calm/intense), Dominance (helpless/in-control)
   - When searching, describe the user's current emotional state in emotional_context
   - Example: "anxious and overwhelmed" or "proud and excited"
   - This shifts which memories surface — same query with different emotions returns different results

5. CONSOLIDATE periodically:
   - Call memory_consolidate after long sessions to strengthen important memories and decay weak ones
"""


class EAAMServer:
    """JSON-RPC 2.0 MCP server over stdio."""

    def __init__(self, config: EAAMConfig | None = None):
        self.config = config or EAAMConfig.load()

        logger.info("EAAM: initializing memory store...")
        self.store = MemoryStore(self.config)
        self.emotion_encoder = EmotionEncoder(self.config.emotion)
        self.encoding_pipeline = EncodingPipeline(self.store, self.emotion_encoder, self.config)
        self.retriever = AssociativeRetriever(self.store, self.emotion_encoder, self.config.retrieval)
        self.consolidator = ConsolidationEngine(self.store, self.config.consolidation)

        self._tools = self._define_tools()
        logger.info("EAAM: ready (%d memories loaded)", self.store.graph.count())

    def _define_tools(self) -> dict[str, dict]:
        return {
            "memory_store": {
                "description": (
                    "Store a memory with automatic emotion detection. "
                    "USE THIS when the user shares experiences, problems, wins, fears, decisions, or anything emotionally meaningful. "
                    "The system detects emotions (valence, arousal, dominance) and builds associations to existing memories. "
                    "You should store important moments PROACTIVELY without being asked."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The text to memorize. Include enough context for future retrieval.",
                        },
                        "conversation_id": {
                            "type": "string",
                            "description": "Group memories from the same conversation. Use a consistent ID per session.",
                        },
                        "topic": {
                            "type": "string",
                            "description": "Topic label (e.g., 'debugging', 'career', 'health')",
                        },
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant"],
                            "description": "Who said this — the user or the assistant",
                        },
                    },
                    "required": ["content"],
                },
            },
            "memory_search": {
                "description": (
                    "Search memories using emotion-anchored associative retrieval. "
                    "USE THIS before answering when past context might help, or when the user references previous conversations. "
                    "IMPORTANT: include emotional_context describing the user's current mood — this shifts which memories surface. "
                    "Same query with 'frustrated' vs 'excited' returns DIFFERENT memories. "
                    "The system searches 4 pathways: semantic similarity, emotional resonance, graph spreading, and involuntary recall."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for. Can be a topic, question, or description of a situation.",
                        },
                        "emotional_context": {
                            "type": "string",
                            "description": "The user's current emotional state. Examples: 'frustrated and stuck', 'excited and proud', 'anxious about deadline', 'calm and reflective'. This biases retrieval toward emotionally matching memories.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            "memory_associative_walk": {
                "description": (
                    "Follow association chains from a memory — 'Proust mode'. "
                    "Start from one memory and walk through connected memories via semantic, emotional, or temporal links. "
                    "USE THIS to explore how memories connect across different topics and time periods."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "start_memory_id": {
                            "type": "string",
                            "description": "ID of the memory to start from (from a previous search result)",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "How many hops to follow (default 3)",
                            "default": 3,
                        },
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["semantic", "emotional", "temporal", "causal", "thematic", "reflection"]},
                            "description": "Filter to specific association types. 'emotional' finds cross-domain connections.",
                        },
                    },
                    "required": ["start_memory_id"],
                },
            },
            "memory_emotional_landscape": {
                "description": (
                    "View the emotional distribution of all stored memories. "
                    "Shows: positive/negative/neutral counts, average arousal, top activated memories. "
                    "USE THIS to understand the user's overall emotional patterns over time."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            "memory_consolidate": {
                "description": (
                    "Run a consolidation cycle — like sleep for memories. "
                    "Decays old unused memories, strengthens frequently accessed ones, "
                    "finds emotional clusters, and creates reflection summaries. "
                    "USE THIS after long sessions or periodically to maintain memory health."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "generate_reflections": {
                            "type": "boolean",
                            "description": "Create summary nodes from emotional clusters (default true)",
                            "default": True,
                        },
                    },
                },
            },
            "memory_stats": {
                "description": "Get memory store statistics: total memories, edges, emotional distribution, average activation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    def run(self):
        """Run the MCP server on stdio."""
        logger.info("EAAM MCP server starting on stdio...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response is not None:
                    self._send(response)
            except json.JSONDecodeError:
                self._send(self._error_response(None, -32700, "Parse error"))
            except Exception as e:
                logger.exception("Unhandled error")
                self._send(self._error_response(None, -32603, str(e)))

    def _handle_request(self, request: dict) -> dict | None:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return self._response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False},
                },
                "serverInfo": {"name": "eaam", "version": "0.1.0"},
            })

        if method.startswith("notifications/"):
            return None

        if method == "tools/list":
            tools = [
                {"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]}
                for name, spec in self._tools.items()
            ]
            return self._response(req_id, {"tools": tools})

        if method == "resources/list":
            return self._response(req_id, {
                "resources": [
                    {
                        "uri": "eaam://system-prompt",
                        "name": "EAAM Memory System Prompt",
                        "description": "Instructions for how to use emotion-anchored memory",
                        "mimeType": "text/plain",
                    }
                ],
            })

        if method == "resources/read":
            uri = params.get("uri", "")
            if uri == "eaam://system-prompt":
                return self._response(req_id, {
                    "contents": [
                        {
                            "uri": "eaam://system-prompt",
                            "mimeType": "text/plain",
                            "text": EAAM_SYSTEM_PROMPT,
                        }
                    ],
                })
            return self._error_response(req_id, -32602, f"Unknown resource: {uri}")

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result = self._call_tool(tool_name, arguments)
                return self._response(req_id, {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                })
            except Exception as e:
                return self._response(req_id, {
                    "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                    "isError": True,
                })

        return self._error_response(req_id, -32601, f"Unknown method: {method}")

    def _call_tool(self, name: str, args: dict) -> Any:
        if name == "memory_store":
            if not args.get("content", "").strip():
                return {"error": "Content cannot be empty"}
            memory = self.encoding_pipeline.encode(
                content=args["content"],
                conversation_id=args.get("conversation_id", ""),
                role=args.get("role", "user"),
                topic=args.get("topic", ""),
            )
            return {
                "stored": True,
                "id": memory.id,
                "emotion_detected": {
                    "valence": round(memory.emotion.valence, 3),
                    "arousal": round(memory.emotion.arousal, 3),
                    "dominance": round(memory.emotion.dominance, 3),
                },
                "activation": round(memory.base_activation, 3),
                "associations_built": len(self.store.graph.get_outgoing_edges(memory.id)),
                "total_memories": self.store.graph.count(),
            }

        if name == "memory_search":
            results = self.retriever.retrieve(
                query=args["query"],
                k=args.get("k", 5),
                emotional_context=args.get("emotional_context"),
            )
            if not results:
                return {"results": [], "message": "No memories found. Store some first with memory_store."}
            return {
                "results": [
                    {
                        "id": r.memory.id,
                        "content": r.memory.content,
                        "relevance": round(r.score, 3),
                        "emotion": {
                            "valence": round(r.memory.emotion.valence, 3),
                            "arousal": round(r.memory.emotion.arousal, 3),
                            "dominance": round(r.memory.emotion.dominance, 3),
                        },
                        "retrieved_via": (
                            "semantic" if r.semantic_score > 0 else
                            "emotional_resonance" if r.emotional_score > 0 else
                            "spreading_activation" if r.spreading_score > 0 else
                            "involuntary_recall"
                        ),
                        "topic": r.memory.topic,
                        "role": r.memory.role,
                    }
                    for r in results
                ],
                "query_emotion_detected": args.get("emotional_context", "not specified"),
            }

        if name == "memory_associative_walk":
            results = self.retriever.associative_walk(
                start_memory_id=args["start_memory_id"],
                max_depth=args.get("max_depth", 3),
                edge_types=args.get("edge_types"),
            )
            return [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "connection_strength": round(r.score, 3),
                    "emotion": {
                        "valence": round(r.memory.emotion.valence, 3),
                        "arousal": round(r.memory.emotion.arousal, 3),
                    },
                    "path": r.path,
                }
                for r in results
            ]

        if name == "memory_emotional_landscape":
            return self.consolidator.get_emotional_landscape()

        if name == "memory_consolidate":
            return self.consolidator.run(
                generate_reflections=args.get("generate_reflections", True),
            )

        if name == "memory_stats":
            return self.store.stats()

        return {"error": f"Unknown tool: {name}"}

    @staticmethod
    def _response(req_id: Any, result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error_response(req_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _send(response: dict):
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
