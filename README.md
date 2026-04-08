# EAAM — Emotion-Anchored Associative Memory

Biologically-inspired memory system for AI that mirrors how human brains encode, associate, and retrieve memories through emotional anchoring.

**Works as an MCP server** for Claude Code, Claude Desktop, Cursor, or any MCP-compatible client. Also runs as a transparent proxy for Ollama / LM Studio.

## Quick Install

```bash
git clone https://github.com/XoAnonXo/eaam.git
cd eaam
pip install -e .
```

## Use as MCP Server (Claude Code / Cursor)

Add to your MCP config:

**Claude Code** (`~/.claude.json`):
```json
{
  "mcpServers": {
    "eaam": {
      "command": "eaam",
      "args": ["serve", "--mode", "mcp"]
    }
  }
}
```

**Cursor** (Settings > MCP Servers):
```json
{
  "eaam": {
    "command": "eaam",
    "args": ["serve", "--mode", "mcp"]
  }
}
```

The AI gets 6 memory tools:

| Tool | What it does |
|---|---|
| `memory_store` | Save with automatic emotional tagging (VAD vectors) |
| `memory_search` | Multi-pathway retrieval: semantic + emotional + spreading + involuntary |
| `memory_associative_walk` | Follow association chains — "Proust mode" |
| `memory_emotional_landscape` | Visualize the emotional distribution of all memories |
| `memory_consolidate` | Run decay / strengthen / cluster / reflect cycle |
| `memory_stats` | Store statistics |

## Use as Ollama Proxy

```bash
# Start EAAM in front of Ollama
eaam serve --mode proxy --upstream http://localhost:11434 --port 8800

# Point your client at localhost:8800 instead of 11434
# Every conversation is automatically encoded with emotions
# Relevant memories are injected into the system prompt
```

## CLI

```bash
eaam init                          # create config at ~/.eaam/config.yaml
eaam store "The server crashed"    # store a memory
eaam search "database problems"    # multi-pathway retrieval
eaam search "problems" -e "frustrated"  # with emotional context
eaam walk <memory_id>              # associative walk (Proust mode)
eaam landscape                     # emotional distribution
eaam consolidate                   # run consolidation cycle
eaam stats                         # store statistics
```

## What Makes This Different

Standard AI memory (RAG) retrieves by semantic similarity alone. EAAM retrieves across **four independent pathways simultaneously**, mimicking human brain circuits:

| Pathway | Brain Analog | What it does |
|---|---|---|
| **Hippocampal** | Hippocampus | Standard semantic search |
| **Amygdalar** | Amygdala | Mood-congruent recall — same feelings, different topics |
| **Spreading** | Cortical association | Graph traversal through multi-typed edges |
| **Involuntary** | Proust effect | High-arousal emotional outliers surface with weak cues |

Each pathway runs independently and results are merged with diversity guarantees — emotional and involuntary memories always get representation, just like in human cognition.

## Benchmark Results

Tested across 10 query scenarios in 5 categories:

| Category | RAG | EAAM | Winner |
|---|---|---|---|
| Emotional Congruence | 0.400 | **0.833** | EAAM +108% |
| Mood-Congruent Recall | 0.449 | **0.793** | EAAM +77% |
| Activation Dynamics | 0.657 | **0.933** | EAAM +42% |
| Semantic Baseline | **0.800** | 0.750 | RAG +6% |
| Cross-Domain Leaps | **0.754** | 0.737 | Nearly tied |
| **Overall** | 0.622 | **0.790** | **EAAM +27%** |

In a dialog benchmark (8 weeks of simulated conversations, 10 test queries):
- Vanilla AI (no memory): **3.6%**
- AI + EAAM: **76.4%** (+2002%)

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Emotion     │     │  Encoding        │     │  Association │
│  Encoder     │────>│  Pipeline        │────>│  Builder     │
│  (VAD tags)  │     │                  │     │  (5 types)   │
└──────────────┘     └──────────────────┘     └──────────────┘
                                                     │
                     ┌──────────────────┐            v
                     │  Multi-Pathway   │     ┌──────────────┐
                     │  Retriever       │<--->│  Memory       │
                     │  (4 pathways)    │     │  Graph +     │
                     └──────────────────┘     │  Vector Store│
                                              └──────────────┘
                     ┌──────────────────┐            ^
                     │  Consolidation   │────────────┘
                     │  Engine          │
                     └──────────────────┘
```

### Memory Node

Each memory carries:
- **Content** + semantic embedding (sentence-transformers)
- **VAD emotional signature** — Valence [-1,1], Arousal [0,1], Dominance [0,1]
- **Base activation** — arousal-modulated initial strength
- **Decay dynamics** — time-based decay, retrieval strengthening, reconsolidation

### Association Edges (5 types)

| Edge Type | What it captures |
|---|---|
| `semantic` | Similar content (embedding cosine similarity) |
| `emotional` | Same emotional tone, different topics |
| `temporal` | Same conversation or time window |
| `causal` | One event led to another |
| `reflection` | Consolidation node to source memories |

### Emotion Detection

Uses `j-hartmann/emotion-english-distilroberta-base` (82M params, runs on CPU) for 7-class emotion detection, mapped to continuous VAD vectors via the NRC VAD Lexicon. Falls back to keyword-based lexicon if the transformer model is unavailable.

## Configuration

```bash
eaam init  # creates ~/.eaam/config.yaml
```

Key parameters (optimized through 1000 automated iterations):

```yaml
retrieval:
  spreading_hops: 1           # single hop, no saturation
  hop_decay: 0.675            # strong per-hop energy
  fan_out_limit: 8            # wide graph traversal
  spreading_cap: 0.85         # generous ceiling
  reconsolidation_rate: 0.07  # memories absorb context on recall
```

## Theoretical Basis

EAAM implements computational analogs of:
- **Spreading activation** (Collins & Loftus, 1975)
- **Emotional memory modulation** (amygdala-hippocampal interaction)
- **Encoding specificity** (Tulving, 1973)
- **Memory reconsolidation** (retrieval strengthens/modifies memories)
- **Sleep consolidation** (decay, strengthen, abstract)
- **Somatic markers** (Damasio) — emotional signatures as retrieval cues
- **Involuntary memory / Proust effect** — high-arousal hubs surface with weak cues

## Self-Improvement Loop

Includes a self-improvement engine inspired by [xoanonxoloop](https://github.com/XoAnonXo/xoanonxoloop):
- Surface-bounded parameter mutations with declared invariants
- Fingerprinted ledger preventing duplicate attempts
- Multi-gate validation (scope -> behavioral -> coherence)
- Cooldown on unproductive dimensions
- Strategy library: algorithmic, structural, formula, compound mutations

Run optimization:
```bash
python3 tests/run_xo_loop.py 100  # 100 iterations of self-improvement
```

## License

MIT
