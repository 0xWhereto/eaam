"""CLI entry point for EAAM."""

from __future__ import annotations

import json
import logging
import sys

import click

from eaam.config import EAAMConfig


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx, config, verbose):
    """EAAM — Emotion-Anchored Associative Memory for AI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["config"] = EAAMConfig.load(config)


@main.command()
@click.option("--mode", type=click.Choice(["mcp", "proxy"]), required=True, help="Server mode")
@click.option("--host", default=None, help="Host to bind (proxy mode)")
@click.option("--port", default=None, type=int, help="Port to bind")
@click.option("--upstream", default=None, help="Upstream LLM URL (proxy mode)")
@click.pass_context
def serve(ctx, mode, host, port, upstream):
    """Start the EAAM server."""
    config: EAAMConfig = ctx.obj["config"]

    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if upstream:
        config.server.upstream_url = upstream

    if mode == "mcp":
        from eaam.server.mcp_server import EAAMServer

        server = EAAMServer(config)
        server.run()
    else:
        from eaam.server.proxy import run_proxy

        click.echo(f"Starting EAAM proxy on {config.server.host}:{config.server.port}")
        click.echo(f"Upstream: {config.server.upstream_url}")
        click.echo("Point your client at this proxy instead of the upstream LLM.")
        run_proxy(config)


@main.command()
@click.argument("text")
@click.option("--conversation", "-C", default="", help="Conversation ID")
@click.option("--topic", "-t", default="", help="Topic label")
@click.pass_context
def store(ctx, text, conversation, topic):
    """Store a memory from the command line."""
    config: EAAMConfig = ctx.obj["config"]

    from eaam.emotion.encoder import EmotionEncoder
    from eaam.engine.encoder import EncodingPipeline
    from eaam.store.memory_store import MemoryStore

    ms = MemoryStore(config)
    ee = EmotionEncoder(config.emotion)
    pipeline = EncodingPipeline(ms, ee, config)

    memory = pipeline.encode(content=text, conversation_id=conversation, topic=topic)

    click.echo(json.dumps({
        "id": memory.id,
        "emotion": memory.emotion.to_dict(),
        "activation": memory.base_activation,
        "edges": len(ms.graph.get_outgoing_edges(memory.id)),
    }, indent=2))


@main.command()
@click.argument("query")
@click.option("--k", default=5, help="Number of results")
@click.option("--emotion", "-e", default=None, help="Emotional context description")
@click.pass_context
def search(ctx, query, k, emotion):
    """Search memories with spreading activation."""
    config: EAAMConfig = ctx.obj["config"]

    from eaam.emotion.encoder import EmotionEncoder
    from eaam.engine.retriever import AssociativeRetriever
    from eaam.store.memory_store import MemoryStore

    ms = MemoryStore(config)
    ee = EmotionEncoder(config.emotion)
    retriever = AssociativeRetriever(ms, ee, config.retrieval)

    results = retriever.retrieve(query=query, k=k, emotional_context=emotion)

    for i, r in enumerate(results):
        click.echo(f"\n--- [{i+1}] Score: {r.score:.4f} ---")
        click.echo(f"  Content: {r.memory.content[:150]}")
        click.echo(f"  Emotion: V={r.memory.emotion.valence:.2f} A={r.memory.emotion.arousal:.2f} D={r.memory.emotion.dominance:.2f}")
        click.echo(f"  Breakdown: sem={r.semantic_score:.3f} emo={r.emotional_score:.3f} act={r.activation_score:.3f} spread={r.spreading_score:.3f}")
        click.echo(f"  Path: {' -> '.join(r.path[:4])}" + (" ..." if len(r.path) > 4 else ""))


@main.command()
@click.option("--no-reflections", is_flag=True, help="Skip reflection generation")
@click.pass_context
def consolidate(ctx, no_reflections):
    """Run a memory consolidation cycle."""
    config: EAAMConfig = ctx.obj["config"]

    from eaam.engine.consolidator import ConsolidationEngine
    from eaam.store.memory_store import MemoryStore

    ms = MemoryStore(config)
    engine = ConsolidationEngine(ms, config.consolidation)
    stats = engine.run(generate_reflections=not no_reflections)

    click.echo(json.dumps(stats, indent=2))


@main.command()
@click.pass_context
def stats(ctx):
    """Show memory store statistics."""
    config: EAAMConfig = ctx.obj["config"]

    from eaam.store.memory_store import MemoryStore

    ms = MemoryStore(config)
    s = ms.stats()
    click.echo(json.dumps(s, indent=2))


@main.command()
@click.pass_context
def landscape(ctx):
    """Show the emotional landscape of stored memories."""
    config: EAAMConfig = ctx.obj["config"]

    from eaam.engine.consolidator import ConsolidationEngine
    from eaam.store.memory_store import MemoryStore

    ms = MemoryStore(config)
    engine = ConsolidationEngine(ms, config.consolidation)
    data = engine.get_emotional_landscape()

    click.echo(json.dumps(data, indent=2, default=str))


@main.command()
@click.argument("memory_id")
@click.option("--depth", default=3, help="Max hops to follow")
@click.pass_context
def walk(ctx, memory_id, depth):
    """Associative walk from a memory — Proust mode."""
    config: EAAMConfig = ctx.obj["config"]

    from eaam.emotion.encoder import EmotionEncoder
    from eaam.engine.retriever import AssociativeRetriever
    from eaam.store.memory_store import MemoryStore

    ms = MemoryStore(config)
    ee = EmotionEncoder(config.emotion)
    retriever = AssociativeRetriever(ms, ee, config.retrieval)

    results = retriever.associative_walk(start_memory_id=memory_id, max_depth=depth)

    if not results:
        click.echo("No associations found from this memory.")
        return

    for i, r in enumerate(results):
        indent = "  " * (len(r.path) - 1)
        click.echo(f"{indent}[{r.score:.3f}] {r.memory.content[:100]}")
        click.echo(f"{indent}  V={r.memory.emotion.valence:.2f} A={r.memory.emotion.arousal:.2f} D={r.memory.emotion.dominance:.2f}")


@main.command()
@click.pass_context
def init(ctx):
    """Initialize EAAM config file."""
    config: EAAMConfig = ctx.obj["config"]
    config.save()
    click.echo(f"Config saved. Edit at: ~/.eaam/config.yaml")


if __name__ == "__main__":
    main()
