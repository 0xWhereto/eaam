"""MCP Server for EAAM — exposes emotion-anchored memory as tools.

Run with: eaam serve --mode mcp
Or configure in Claude Desktop / Cursor MCP settings.
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


class EAAMServer:
    """JSON-RPC 2.0 MCP server over stdio."""

    def __init__(self, config: EAAMConfig | None = None):
        self.config = config or EAAMConfig.load()

        # Initialize components
        self.store = MemoryStore(self.config)
        self.emotion_encoder = EmotionEncoder(self.config.emotion)
        self.encoding_pipeline = EncodingPipeline(self.store, self.emotion_encoder, self.config)
        self.retriever = AssociativeRetriever(self.store, self.emotion_encoder, self.config.retrieval)
        self.consolidator = ConsolidationEngine(self.store, self.config.consolidation)

        self._tools = self._define_tools()
        self._running = True

    def _define_tools(self) -> dict[str, dict]:
        return {
            "memory_store": {
                "description": "Store a new memory with automatic emotional tagging and association building. "
                "The system detects emotions in the text, assigns a VAD (Valence-Arousal-Dominance) vector, "
                "and builds semantic, emotional, and temporal associations to existing memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The text content to memorize"},
                        "conversation_id": {"type": "string", "description": "ID to group memories from the same conversation"},
                        "topic": {"type": "string", "description": "Optional topic label"},
                        "role": {"type": "string", "enum": ["user", "assistant"], "description": "Who said this"},
                    },
                    "required": ["content"],
                },
            },
            "memory_search": {
                "description": "Retrieve memories using spreading activation with emotional congruence. "
                "Unlike simple vector search, this propagates through the association graph, "
                "boosting memories that match the current emotional context — even if they're about different topics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "emotional_context": {"type": "string", "description": "Description of current emotional tone (e.g., 'frustrated', 'curious and excited')"},
                        "k": {"type": "integer", "description": "Number of results to return", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            "memory_associative_walk": {
                "description": "Walk the association graph from a starting memory — 'Proust mode'. "
                "Follows the strongest edges outward, showing how memories connect across domains "
                "through emotional and semantic links.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "start_memory_id": {"type": "string", "description": "ID of the memory to start from"},
                        "max_depth": {"type": "integer", "description": "How many hops to follow", "default": 3},
                        "edge_types": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["semantic", "emotional", "temporal", "causal", "thematic", "reflection"]},
                            "description": "Filter to specific edge types",
                        },
                    },
                    "required": ["start_memory_id"],
                },
            },
            "memory_emotional_landscape": {
                "description": "Get the emotional distribution and activation landscape of all stored memories. "
                "Shows positive/negative/neutral distribution, arousal levels, and top activated memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            "memory_consolidate": {
                "description": "Trigger a consolidation cycle — decay weak memories, strengthen frequently accessed ones, "
                "find emotional clusters, create reflection nodes, and prune weak edges. "
                "Mirrors human sleep consolidation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "generate_reflections": {"type": "boolean", "description": "Whether to create reflection nodes from clusters", "default": True},
                    },
                },
            },
            "memory_stats": {
                "description": "Get statistics about the memory store — node count, edge count, edge type distribution, average activation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    def run(self):
        """Run the MCP server on stdio."""
        logger.info("EAAM MCP server starting...")
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

    def _handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})

        if method == "initialize":
            return self._response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "eaam", "version": "0.1.0"},
            })

        if method.startswith("notifications/"):
            return None  # notifications get no response per JSON-RPC 2.0

        if method == "tools/list":
            tools = [
                {"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]}
                for name, spec in self._tools.items()
            ]
            return self._response(req_id, {"tools": tools})

        if method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = self._call_tool(tool_name, arguments)
            return self._response(req_id, {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            })

        return self._error_response(req_id, -32601, f"Unknown method: {method}")

    def _call_tool(self, name: str, args: dict) -> Any:
        if name == "memory_store":
            memory = self.encoding_pipeline.encode(
                content=args["content"],
                conversation_id=args.get("conversation_id", ""),
                role=args.get("role", "user"),
                topic=args.get("topic", ""),
            )
            return {
                "id": memory.id,
                "emotion": memory.emotion.to_dict(),
                "base_activation": memory.base_activation,
                "associations_built": len(self.store.graph.get_outgoing_edges(memory.id)),
            }

        if name == "memory_search":
            results = self.retriever.retrieve(
                query=args["query"],
                k=args.get("k", 5),
                emotional_context=args.get("emotional_context"),
            )
            return [
                {
                    "id": r.memory.id,
                    "content": r.memory.content,
                    "score": round(r.score, 4),
                    "breakdown": {
                        "semantic": round(r.semantic_score, 4),
                        "emotional": round(r.emotional_score, 4),
                        "activation": round(r.activation_score, 4),
                        "spreading": round(r.spreading_score, 4),
                    },
                    "emotion": r.memory.emotion.to_dict(),
                    "path_length": len(r.path),
                }
                for r in results
            ]

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
                    "score": round(r.score, 4),
                    "emotion": r.memory.emotion.to_dict(),
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
