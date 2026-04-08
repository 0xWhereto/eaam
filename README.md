# EAAM — Emotion-Anchored Associative Memory

> Give any AI a memory that feels. EAAM tags every memory with emotions, builds associative links across topics, and retrieves based on how you feel — not just what you said.

**Install as an MCP server** and any AI (Claude Code, Cursor, Claude Desktop, Windsurf) gains persistent, emotionally-aware long-term memory that works the way human memory does.

---

## Why EAAM Exists

Current AI has no memory between conversations. Ask it something Monday, and by Tuesday it's forgotten. Memory solutions like RAG exist, but they only search by keyword/topic similarity — if you ask about "database issues," you get database results. Period.

Human memory doesn't work this way. When you're panicking about a server crash, your brain also surfaces that time you panicked during a job interview, or when you got scary health news — because the **emotion is the same**, even though the topics are completely different. Your mood colors what you remember. A frustrating day makes frustrating memories more accessible. A joyful moment brings back other joys.

EAAM brings this to AI. Every memory is tagged with an emotional signature. Associations are built not just by topic but by shared emotional tone. Retrieval uses four independent brain-inspired pathways. The result: an AI that remembers not just *what* happened, but *how it felt* — and uses that to surface the right context at the right time.

---

## What Makes EAAM Different

### Standard RAG vs EAAM

| Feature | Standard RAG | EAAM |
|---|---|---|
| Retrieval method | Semantic similarity only | 4 independent pathways |
| Emotional awareness | None | Auto-detects emotions via transformer model |
| Same query, different mood | Same results always | Different results based on emotional state |
| Cross-domain recall | Only if topics match | Finds memories with same emotional tone across any domain |
| Involuntary memories | Impossible | High-arousal memories surface with even weak cues |
| Memory consolidation | None | Decay, strengthen, cluster, reflect — like sleep |
| Reconsolidation | None | Memories absorb current context when recalled |

### The Four Retrieval Pathways

EAAM doesn't use a single scoring function. It runs **four independent brain-inspired circuits** in parallel:

| Pathway | Inspired By | What It Does |
|---|---|---|
| **Hippocampal** | Hippocampus | Classic semantic search — finds topically similar memories |
| **Amygdalar** | Amygdala | Mood-congruent recall — when you're anxious, anxious memories surface regardless of topic |
| **Spreading Activation** | Cortical networks | Follows association chains through the memory graph — one memory activates its neighbors |
| **Involuntary** | Proust effect | Extreme emotional memories surface with even weak cues — completely "out of context" recall |

Each pathway contributes independently, and results are merged with diversity guarantees. This means emotional and involuntary memories always get representation — you can't suppress them, just like in a real brain.

### Emotion Detection

Every memory is tagged with a 3-dimensional emotional vector (VAD):
- **Valence** [-1, 1]: negative to positive
- **Arousal** [0, 1]: calm to intense
- **Dominance** [0, 1]: helpless to in-control

Detected automatically using a transformer model (`j-hartmann/emotion-english-distilroberta-base`, 82M params, runs on CPU). No configuration needed — just speak naturally and the system understands the emotional tone.

### Tone and Context Awareness

When you search memory, you describe the current emotional context:
- `"frustrated and stuck"` surfaces past frustrations, debugging struggles, incidents
- `"excited and proud"` surfaces past wins, celebrations, breakthroughs
- Same query, completely different results — just like human mood-congruent recall

---

## Benchmark Results

### EAAM vs Standard RAG (10 scenarios, 5 categories)

| Category | RAG | EAAM | Winner |
|---|---|---|---|
| Emotional Congruence | 0.400 | **0.833** | **EAAM +108%** |
| Mood-Congruent Recall | 0.449 | **0.793** | **EAAM +77%** |
| Activation Dynamics | 0.657 | **0.933** | **EAAM +42%** |
| Semantic Baseline | **0.800** | 0.750 | RAG +6% |
| Cross-Domain Leaps | **0.754** | 0.737 | Nearly tied |
| **Overall** | 0.622 | **0.790** | **EAAM +27%** |

### Dialog Benchmark (8 weeks of simulated conversations, 10 test queries)

| Agent | Score |
|---|---|
| Vanilla AI (no memory) | 3.6% |
| **AI + EAAM** | **76.4%** (+2002%) |

Categories tested: direct recall, emotional association, cross-domain leaps, pattern recognition, empathy.

### Self-Improvement

EAAM includes an autonomous optimization loop (inspired by [xoanonxoloop](https://github.com/XoAnonXo/xoanonxoloop)):
- 10 parallel agents explored 1,000 total parameter configurations
- Surface-bounded mutations with multi-gate validation
- Best agent achieved **+27% over RAG** through 7 structural discoveries

---

## Setup

### Requirements

- Python 3.9+
- ~2GB disk space (for transformer model, downloaded once)
- Works on CPU — no GPU needed

### Install

```bash
git clone https://github.com/0xWhereto/eaam.git
cd eaam
pip install -e .
```

First run will download the emotion detection model (~300MB). After that, everything runs locally with zero cloud dependencies.

### Setup with Claude Code

Add to your Claude Code MCP config (`~/.claude.json` or project-level `.claude/settings.json`):

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

Restart Claude Code. You'll see EAAM's 6 memory tools become available. Claude will start using them proactively.

### Setup with Claude Desktop

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

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

### Setup with Cursor

Go to **Settings > MCP Servers** and add:

```json
{
  "eaam": {
    "command": "eaam",
    "args": ["serve", "--mode", "mcp"]
  }
}
```

### Setup with Ollama / LM Studio (Proxy Mode)

For local models, EAAM runs as a transparent proxy that intercepts every conversation:

```bash
# Start EAAM proxy
eaam serve --mode proxy --upstream http://localhost:11434 --port 8800

# Point your client at localhost:8800 instead of 11434
# Memory encoding and retrieval happens automatically
```

### Verify Installation

```bash
# Check CLI works
eaam --help

# Store a test memory
eaam store "Testing EAAM — this is a happy moment"

# Search with emotional context
eaam search "testing" -e "excited"

# View memory stats
eaam stats
```

---

## How It Works

### Architecture

```
User speaks
    │
    v
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│  Emotion     │     │  Encoding        │     │  Association │
│  Encoder     │────>│  Pipeline        │────>│  Builder     │
│ (auto VAD)   │     │                  │     │ (5 edge      │
└──────────────┘     └──────────────────┘     │  types)      │
                                              └──────┬───────┘
                                                     │
AI asks a question                                   v
    │                ┌──────────────────┐     ┌──────────────┐
    │                │  Multi-Pathway   │     │  Memory      │
    └───────────────>│  Retriever       │<───>│  Graph +     │
                     │ (4 pathways)     │     │  Vectors     │
                     └──────────────────┘     └──────────────┘
                             │
                             v
                     Relevant memories returned
                     with emotional context
                             │
                             v
                     AI reasons over them
                     and responds
```

### Memory Storage

Each memory stored in EAAM carries:

- **Content** — the raw text, embedded as a semantic vector
- **Emotional signature** — Valence, Arousal, Dominance (auto-detected)
- **Activation level** — modulated by arousal at encoding (intense moments = stronger memories)
- **Associations** — 5 types of weighted edges connecting to other memories

### Association Types

| Edge | What It Captures | Why It Matters |
|---|---|---|
| `semantic` | Topic similarity | "database crash" links to "database migration" |
| `emotional` | Shared emotional tone, different topics | "panicking about server" links to "panicking about interview" |
| `temporal` | Same conversation or time window | Memories created together stay linked |
| `causal` | One event led to another | "bug found" links to "bug fixed" |
| `reflection` | Consolidation summary to source memories | Higher-order patterns emerge |

### Consolidation (Memory Sleep)

Run `eaam consolidate` or call `memory_consolidate` to:
1. **Decay** — weaken unused memories over time
2. **Strengthen** — boost frequently retrieved, high-arousal memories
3. **Cluster** — find groups of memories with similar emotional signatures
4. **Reflect** — generate summary nodes from emotional clusters
5. **Prune** — remove weak association edges

This mirrors what human brains do during sleep.

### Reconsolidation

When a memory is retrieved, it slightly absorbs the current emotional context (5% shift toward current state). Over time, this means memories evolve — just like human memories, which are reconstructed slightly differently each time they're recalled.

---

## MCP Tools Reference

| Tool | When to Use | Key Parameters |
|---|---|---|
| `memory_store` | User shares something meaningful | `content`, `topic`, `conversation_id` |
| `memory_search` | Need past context to answer well | `query`, `emotional_context`, `k` |
| `memory_associative_walk` | Explore how memories connect | `start_memory_id`, `max_depth`, `edge_types` |
| `memory_emotional_landscape` | Understand user's emotional patterns | (none) |
| `memory_consolidate` | After long sessions | `generate_reflections` |
| `memory_stats` | Check memory health | (none) |

### Example: `memory_search`

```json
{
  "query": "deployment problems",
  "emotional_context": "frustrated and anxious",
  "k": 5
}
```

Returns memories ranked by multi-pathway scoring. The `emotional_context` parameter shifts results — the same query with `"proud and relieved"` returns different memories.

---

## CLI Reference

```bash
eaam init                              # create ~/.eaam/config.yaml
eaam store "text to remember"          # store with auto emotion detection
eaam search "query" -e "mood"          # search with emotional context
eaam walk <memory_id> --depth 3        # follow association chains
eaam landscape                         # emotional distribution
eaam consolidate                       # run memory sleep cycle
eaam stats                             # store statistics
eaam serve --mode mcp                  # start MCP server
eaam serve --mode proxy --port 8800    # start Ollama proxy
```

---

## Configuration

```bash
eaam init  # creates ~/.eaam/config.yaml
```

Key parameters:

```yaml
emotion:
  model: j-hartmann/emotion-english-distilroberta-base
  use_transformer: true    # false = keyword-only fallback (no model download)

graph:
  persist_path: ~/.eaam/data/graph   # where memory graph is saved

vector:
  persist_path: ~/.eaam/data/vectors  # where embeddings are saved
```

All data is stored locally. Nothing is sent to any cloud service.

---

## Theoretical Foundations

EAAM implements computational analogs of established neuroscience:

| Mechanism | Neuroscience Basis | EAAM Implementation |
|---|---|---|
| Spreading activation | Collins & Loftus (1975) | Graph traversal through weighted multi-type edges |
| Emotional memory modulation | Amygdala-hippocampal interaction | Arousal-modulated encoding strength |
| Encoding specificity | Tulving (1973) | Context vectors stored with each memory |
| Memory reconsolidation | Nader et al. (2000) | Emotional signature shifts on retrieval |
| Sleep consolidation | Walker & Stickgold (2004) | Decay/strengthen/cluster/reflect cycle |
| Somatic markers | Damasio (1996) | VAD emotional signatures as retrieval cues |
| Involuntary memory | Proust effect / Berntsen (2009) | High-activation emotional outlier pathway |
| Mood-congruent recall | Bower (1981) | Amygdalar pathway biases by current emotional state |

---

## Project Structure

```
eaam/
  __init__.py
  cli.py                    # CLI entry point (8 commands)
  config.py                 # Configuration with YAML persistence
  models.py                 # Memory, VAD, Edge, RetrievalResult
  emotion/
    encoder.py              # Transformer + lexicon fallback
    vad_lexicon.py          # NRC VAD mappings (28 emotions)
  store/
    graph.py                # In-memory graph with JSON persistence
    vector.py               # ChromaDB + sentence-transformers
    memory_store.py         # Unified store
  engine/
    encoder.py              # Encoding pipeline + association builder
    retriever.py            # Multi-pathway retrieval (v2)
    consolidator.py         # Memory sleep cycle
    loop_engine.py          # Surface-bounded optimization
    xo_loop.py              # XO-ANON-XO strategy loop
    surfaces.yaml           # Optimization surface definitions
  server/
    mcp_server.py           # MCP server (JSON-RPC 2.0 over stdio)
    proxy.py                # Ollama/OpenAI proxy (FastAPI)
tests/
  test_models.py            # Unit tests (18 passing)
  test_emotion.py
  test_graph.py
  benchmark.py              # RAG vs EAAM benchmark
  dialog_benchmark.py       # Multi-session dialog benchmark
  bench_locomo.py           # LoCoMo MC10 benchmark adapter
```

---

## License

MIT

---

## Acknowledgments

- Emotion detection: [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- VAD lexicon: [NRC VAD Lexicon v2](https://saifmohammad.com/WebPages/nrc-vad.html) (Mohammad, 2018)
- Self-improvement loop: inspired by [xoanonxoloop](https://github.com/XoAnonXo/xoanonxoloop)
- Embeddings: [sentence-transformers](https://www.sbert.net/)
- Vector store: [ChromaDB](https://www.trychroma.com/)
