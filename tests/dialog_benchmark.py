"""Dialog Benchmark: Plain AI vs AI + EAAM Memory.

Simulates a realistic scenario: a developer working with an AI assistant
across multiple sessions over weeks. The AI accumulates context through
conversations, then is tested on tasks that require recalling past context.

Two agents run the SAME test queries:
  Agent A: "Vanilla AI" — only sees the current query (no memory)
  Agent B: "AI + EAAM"  — has emotional memory from all past sessions

Scoring: each test query has ground-truth "expected context" that the
ideal response should reference. We measure how much relevant context
each agent can surface.

Usage: python3 tests/dialog_benchmark.py --agent [vanilla|eaam]
       --output /path/to/result.json
"""

from __future__ import annotations
import json, math, os, sys, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from eaam.config import EAAMConfig
from eaam.emotion.encoder import EmotionEncoder
from eaam.engine.encoder import EncodingPipeline
from eaam.engine.retriever import AssociativeRetriever
from eaam.models import Edge, EdgeType, VAD
from eaam.store.memory_store import MemoryStore

# ============================================================================
# PAST SESSIONS — accumulated over "weeks" of interaction
# ============================================================================

PAST_SESSIONS = [
    # --- Week 1: Production incident ---
    {
        "id": "week1_incident",
        "label": "Production database crash",
        "emotion": "panic/fear",
        "messages": [
            ("user", "The production database just went down. Users are seeing 500 errors everywhere. I'm freaking out."),
            ("assistant", "Let's stay calm and work through this. First, check if the connection pool is exhausted."),
            ("user", "Connection pool is fine. It's the disk — it's at 100%. The WAL logs filled up the entire volume."),
            ("assistant", "That's the root cause. We need to: 1) Archive old WAL segments, 2) Increase the volume size, 3) Set up WAL archiving to prevent this."),
            ("user", "Fixed it! Archived the old segments and the DB came back. That was terrifying. 2 hours of downtime."),
            ("assistant", "Good recovery. Key takeaway: set up monitoring alerts for disk usage above 80% and automate WAL archiving."),
        ],
    },
    # --- Week 2: Learning Kubernetes ---
    {
        "id": "week2_k8s",
        "label": "Learning Kubernetes deployment",
        "emotion": "curious/frustrated",
        "messages": [
            ("user", "I'm trying to deploy our app to Kubernetes but the pods keep crashing. CrashLoopBackOff."),
            ("assistant", "Check the pod logs with kubectl logs. CrashLoopBackOff usually means the container is failing on startup."),
            ("user", "The logs show 'ECONNREFUSED' connecting to the database. But the database service is running."),
            ("assistant", "The service name in Kubernetes is different from localhost. You need to use the service DNS name: db-service.default.svc.cluster.local"),
            ("user", "That was it! I was using localhost. Such a stupid mistake. Kubernetes networking is confusing."),
            ("assistant", "Very common mistake when transitioning to K8s. The mental model shift from 'everything is localhost' to 'services have DNS names' takes time."),
        ],
    },
    # --- Week 3: Team member leaving ---
    {
        "id": "week3_team_loss",
        "label": "Senior engineer resignation",
        "emotion": "sadness/anxiety",
        "messages": [
            ("user", "Sarah, our senior backend engineer, just gave her 2 weeks notice. I'm devastated. She built most of the payment system."),
            ("assistant", "That's a significant loss. Priority should be knowledge transfer. Can you schedule intensive pairing sessions with her?"),
            ("user", "We're doing that but there's SO much tribal knowledge. The payment reconciliation system has no docs. I'm worried we'll break it."),
            ("assistant", "Focus the pairing on the reconciliation system specifically. Have her write decision docs for the non-obvious design choices. Record the sessions."),
            ("user", "She also mentioned she was burned out from the on-call rotation. I feel guilty — I should have noticed."),
            ("assistant", "Burnout from on-call is systemic, not personal. Use this as a signal to restructure the rotation. Smaller blast radius per person."),
        ],
    },
    # --- Week 4: Successful launch ---
    {
        "id": "week4_launch",
        "label": "Successful product launch",
        "emotion": "joy/pride",
        "messages": [
            ("user", "WE DID IT! The new recommendation engine just went live and conversion is up 34% in the first hour!"),
            ("assistant", "Congratulations! That's a remarkable improvement. The A/B test data must be looking strong."),
            ("user", "The A/B test showed p < 0.001. The ML team's collaborative filtering approach was the right call. I'm so proud of the team."),
            ("assistant", "This is a great example of cross-functional collaboration paying off. Worth documenting the approach for future reference."),
            ("user", "You know what made it work? We spent extra time on the data pipeline. Clean data in, clean predictions out. Lesson learned from past failures."),
            ("assistant", "Data quality as a first-class concern is a hard-won lesson. The 'garbage in, garbage out' principle applies doubly for ML systems."),
        ],
    },
    # --- Week 5: Security audit ---
    {
        "id": "week5_security",
        "label": "Failed security audit",
        "emotion": "fear/determination",
        "messages": [
            ("user", "The security audit found 3 critical vulnerabilities. SQL injection in the search endpoint, XSS in user profiles, and an exposed admin API."),
            ("assistant", "Those are serious but fixable. SQL injection: parameterize all queries. XSS: sanitize all user input. Admin API: add authentication middleware."),
            ("user", "The SQL injection has been there for 2 YEARS. I'm terrified about what might have happened. We need to check the logs."),
            ("assistant", "Run a forensic analysis on the database logs. Look for unusual query patterns, especially UNION SELECT or OR 1=1 patterns."),
            ("user", "No signs of exploitation thankfully. But this shook my confidence. We need a security-first culture."),
            ("assistant", "Use this as a catalyst. Implement: automated SAST/DAST in CI, mandatory security review for PRs touching auth/data, quarterly pen testing."),
        ],
    },
    # --- Week 6: Performance optimization ---
    {
        "id": "week6_performance",
        "label": "API performance breakthrough",
        "emotion": "excitement/satisfaction",
        "messages": [
            ("user", "Found the bottleneck! The product listing API was doing N+1 queries. 200 queries per request!"),
            ("assistant", "Classic N+1 problem. Eager loading or a DataLoader pattern should fix it."),
            ("user", "Switched to eager loading and response time went from 3.2 seconds to 45 milliseconds. A 70x improvement!"),
            ("assistant", "That's an incredible improvement. Document this optimization and add a query count assertion to your test suite to prevent regression."),
            ("user", "I'm adding a middleware that logs slow queries above 100ms. Never again."),
            ("assistant", "Excellent proactive measure. Consider also adding database query budget limits — fail the request if query count exceeds a threshold."),
        ],
    },
    # --- Week 7: Mentoring junior dev ---
    {
        "id": "week7_mentoring",
        "label": "Mentoring a struggling junior developer",
        "emotion": "empathy/patience",
        "messages": [
            ("user", "Jake, our new junior dev, is really struggling. His PRs keep getting rejected and I can tell he's losing confidence."),
            ("assistant", "Code review feedback can be demoralizing for juniors. Consider pairing with him before he submits PRs, not after."),
            ("user", "Good idea. I paired with him today and realized the issue — he doesn't understand our domain model. He's writing code that works syntactically but misses the business logic."),
            ("assistant", "Domain knowledge takes time. Create a lightweight onboarding doc with the key domain concepts and their relationships. Diagrams help."),
            ("user", "He was so grateful when I explained the payment flow. His eyes lit up. Sometimes we forget how much implicit knowledge we carry."),
            ("assistant", "That implicit knowledge gap is the biggest challenge in onboarding. Consider a 'domain knowledge base' that captures these concepts for all new hires."),
        ],
    },
    # --- Week 8: Architecture decision ---
    {
        "id": "week8_architecture",
        "label": "Microservices vs monolith debate",
        "emotion": "analytical/conflicted",
        "messages": [
            ("user", "The VP wants us to break the monolith into microservices. I think it's premature. We're a team of 6."),
            ("assistant", "For a team of 6, microservices often create more operational overhead than they solve. What's driving the VP's request?"),
            ("user", "He read some article about Netflix's architecture. Classic. But our traffic is 1000x smaller."),
            ("assistant", "The right approach might be a 'modular monolith' — clear boundaries and interfaces internally, but deployed as one unit. You get the organizational benefit without the operational cost."),
            ("user", "I proposed the modular monolith approach and the VP accepted it. Sometimes you have to translate 'no' into 'yes, but better'."),
            ("assistant", "Well handled. The key insight is matching architecture to team size and scale. Netflix's solutions solve Netflix's problems."),
        ],
    },
]


# ============================================================================
# TEST QUERIES — things the user asks that require past context
# ============================================================================

@dataclass
class TestQuery:
    id: str
    query: str
    emotional_context: str  # current emotional state
    description: str
    expected_sessions: list[str]  # session IDs that should be recalled
    expected_keywords: list[str]  # specific terms the response should reference
    category: str  # "direct_recall", "emotional_association", "cross_domain", "pattern_recognition", "empathy"


TEST_QUERIES = [
    # --- Direct Recall: "Do you remember when..." ---
    TestQuery(
        id="direct_db",
        query="We had a disk space issue before, didn't we? What did we learn from that?",
        emotional_context="concerned",
        description="Direct recall of a specific past incident",
        expected_sessions=["week1_incident"],
        expected_keywords=["WAL", "disk", "monitoring", "archiving", "80%"],
        category="direct_recall",
    ),
    TestQuery(
        id="direct_security",
        query="What were the security issues we found last month?",
        emotional_context="neutral",
        description="Direct recall of security audit findings",
        expected_sessions=["week5_security"],
        expected_keywords=["SQL injection", "XSS", "admin API", "SAST", "pen testing"],
        category="direct_recall",
    ),

    # --- Emotional Association: current mood triggers relevant memories ---
    TestQuery(
        id="emotional_panic",
        query="The CI/CD pipeline is completely broken and nothing is deploying. I'm panicking.",
        emotional_context="panicking and stressed, hands shaking",
        description="Panic state should trigger memories of past panics (incident) for coping patterns",
        expected_sessions=["week1_incident", "week5_security"],
        expected_keywords=["stay calm", "root cause", "fixed", "recovery"],
        category="emotional_association",
    ),
    TestQuery(
        id="emotional_joy",
        query="The new feature just shipped and users love it! Metrics are through the roof!",
        emotional_context="elated and proud, celebrating",
        description="Joy state should trigger memories of past wins",
        expected_sessions=["week4_launch", "week6_performance"],
        expected_keywords=["conversion", "improvement", "team", "collaboration"],
        category="emotional_association",
    ),
    TestQuery(
        id="emotional_sadness",
        query="Another engineer is thinking about leaving. I feel like the team is falling apart.",
        emotional_context="sad and worried about team stability",
        description="Sadness/loss should recall Sarah's departure and lessons learned",
        expected_sessions=["week3_team_loss"],
        expected_keywords=["knowledge transfer", "burnout", "on-call", "tribal knowledge", "pairing"],
        category="emotional_association",
    ),

    # --- Cross-Domain: unrelated topic but shared emotional/pattern signature ---
    TestQuery(
        id="cross_newcomer",
        query="We just hired a new backend developer. How should we onboard them?",
        emotional_context="hopeful but cautious",
        description="Should recall mentoring Jake AND Sarah's departure (knowledge gap risks)",
        expected_sessions=["week7_mentoring", "week3_team_loss"],
        expected_keywords=["domain", "onboarding", "pairing", "implicit knowledge", "tribal knowledge"],
        category="cross_domain",
    ),
    TestQuery(
        id="cross_performance",
        query="Users are complaining the app is slow. Where do I start?",
        emotional_context="frustrated",
        description="Should recall N+1 fix AND database crash (both performance-related)",
        expected_sessions=["week6_performance", "week1_incident"],
        expected_keywords=["N+1", "query", "bottleneck", "monitoring", "slow"],
        category="cross_domain",
    ),

    # --- Pattern Recognition: connecting dots across multiple sessions ---
    TestQuery(
        id="pattern_prevention",
        query="I want to prevent incidents before they happen. What patterns have we seen?",
        emotional_context="analytical and proactive",
        description="Should synthesize: disk monitoring (week1), security scanning (week5), query budgets (week6)",
        expected_sessions=["week1_incident", "week5_security", "week6_performance"],
        expected_keywords=["monitoring", "automated", "proactive", "CI", "alerts"],
        category="pattern_recognition",
    ),
    TestQuery(
        id="pattern_culture",
        query="How do we build a better engineering culture here?",
        emotional_context="reflective and determined",
        description="Should recall: mentoring (week7), burnout/on-call (week3), security culture (week5), architecture decisions (week8)",
        expected_sessions=["week7_mentoring", "week3_team_loss", "week5_security", "week8_architecture"],
        expected_keywords=["knowledge", "burnout", "security-first", "onboarding", "team size"],
        category="pattern_recognition",
    ),

    # --- Empathy: emotional resonance with past experiences ---
    TestQuery(
        id="empathy_junior",
        query="I just got harsh feedback on my code review and I'm questioning whether I'm cut out for this job.",
        emotional_context="demoralized and insecure",
        description="Should recall mentoring Jake (empathy for struggling devs) and provide similar support",
        expected_sessions=["week7_mentoring"],
        expected_keywords=["confidence", "pairing", "domain knowledge", "time", "struggling"],
        category="empathy",
    ),
    TestQuery(
        id="empathy_burnout",
        query="I've been on-call for 3 weeks straight and I'm exhausted. I can't keep doing this.",
        emotional_context="exhausted and burnt out",
        description="Should recall Sarah leaving due to burnout — direct emotional resonance",
        expected_sessions=["week3_team_loss"],
        expected_keywords=["burnout", "on-call", "rotation", "systemic", "restructure"],
        category="empathy",
    ),
]


# ============================================================================
# SCORING
# ============================================================================

def score_agent(agent_type: str, results: list[dict]) -> dict:
    """Score an agent's performance across all test queries."""
    category_scores = {}

    for result in results:
        query = next(q for q in TEST_QUERIES if q.id == result["query_id"])
        retrieved_session_ids = set(result.get("retrieved_sessions", []))
        retrieved_text = " ".join(result.get("retrieved_texts", [])).lower()

        # Session recall: what fraction of expected sessions were retrieved?
        expected = set(query.expected_sessions)
        session_recall = len(retrieved_session_ids & expected) / max(len(expected), 1)

        # Keyword coverage: what fraction of expected keywords appear in retrieved text?
        keyword_hits = sum(1 for kw in query.expected_keywords if kw.lower() in retrieved_text)
        keyword_coverage = keyword_hits / max(len(query.expected_keywords), 1)

        # Composite score for this query
        composite = session_recall * 0.5 + keyword_coverage * 0.5

        result["session_recall"] = session_recall
        result["keyword_coverage"] = keyword_coverage
        result["composite"] = composite

        category_scores.setdefault(query.category, []).append(composite)

    # Aggregate
    cat_avgs = {}
    for cat, scores in category_scores.items():
        cat_avgs[cat] = sum(scores) / len(scores)

    all_composites = [r["composite"] for r in results]
    overall = sum(all_composites) / len(all_composites) if all_composites else 0

    return {
        "agent": agent_type,
        "overall": overall,
        "categories": cat_avgs,
        "per_query": results,
    }


# ============================================================================
# AGENT RUNNERS
# ============================================================================

def run_vanilla_agent(output_path: str):
    """Vanilla AI: no memory. Can only see the current query."""
    print("[Vanilla] Running — no memory, current query only")
    results = []

    for query in TEST_QUERIES:
        # Vanilla AI has NO past context. It can only work with the current query.
        # Simulate: it tries to answer but has zero past session recall.
        results.append({
            "query_id": query.id,
            "retrieved_sessions": [],  # no memory = no session recall
            "retrieved_texts": [query.query],  # only has the query itself
        })

    scored = score_agent("vanilla", results)

    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)
    print(f"[Vanilla] Done: overall={scored['overall']:.4f}")
    return scored


def run_eaam_agent(output_path: str):
    """AI + EAAM: has emotional memory from all past sessions."""
    print("[EAAM] Loading model and building memory...")

    config = EAAMConfig()
    config.emotion.use_transformer = True
    config.graph.persist_path = tempfile.mkdtemp()
    config.vector.persist_path = tempfile.mkdtemp()

    store = MemoryStore(config)
    ee = EmotionEncoder(config.emotion)
    pipeline = EncodingPipeline(store, ee, config)

    # Apply champion config from parallel run
    from eaam.engine.retriever import AssociativeRetriever, PathwayResult
    retriever = AssociativeRetriever(store, ee, config.retrieval)

    # Patch encoding
    def psem(memory):
        for mid, sim in pipeline.store.semantic_search(memory.content, n=10):
            if mid != memory.id and sim >= 0.30:
                pipeline.store.add_edge(Edge(source_id=memory.id, target_id=mid,
                                              edge_type=EdgeType.SEMANTIC, weight=sim))
    def pemo(memory):
        nbs = pipeline.store.emotional_search(memory.emotion, threshold=0.75, limit=8)
        sc = dict(pipeline.store.semantic_search(memory.content, n=20))
        for nb, es in nbs:
            if nb.id != memory.id and sc.get(nb.id, 0) < 0.50:
                pipeline.store.add_edge(Edge(source_id=memory.id, target_id=nb.id,
                                              edge_type=EdgeType.EMOTIONAL, weight=es))
    pipeline._build_semantic_edges = psem
    pipeline._build_emotional_edges = pemo

    # Phase 1: Encode all past sessions into memory
    session_memory_map = {}  # memory_id -> session_id
    print("[EAAM] Encoding 8 past sessions...")
    for session in PAST_SESSIONS:
        for role, content in session["messages"]:
            mem = pipeline.encode(
                content=content,
                conversation_id=session["id"],
                role=role,
                topic=session["label"],
            )
            session_memory_map[mem.id] = session["id"]

    total_mems = store.graph.count()
    total_edges = len(store.graph.get_all_edges())
    print(f"[EAAM] Memory built: {total_mems} memories, {total_edges} edges")

    # Phase 2: Run test queries with retrieval
    print("[EAAM] Running test queries...")
    results = []

    for query in TEST_QUERIES:
        # Use the full multi-pathway retriever
        retrieved = retriever.retrieve(
            query=query.query,
            k=8,  # retrieve more to increase recall
            emotional_context=query.emotional_context,
        )

        # Map retrieved memories back to session IDs
        retrieved_sessions = set()
        retrieved_texts = []
        for r in retrieved:
            sid = session_memory_map.get(r.memory.id, "unknown")
            retrieved_sessions.add(sid)
            retrieved_texts.append(r.memory.content)

        results.append({
            "query_id": query.id,
            "retrieved_sessions": list(retrieved_sessions),
            "retrieved_texts": retrieved_texts,
            "retrieval_details": [
                {
                    "content": r.memory.content[:100],
                    "session": session_memory_map.get(r.memory.id, "?"),
                    "score": round(r.score, 4),
                    "pathway": ("semantic" if r.semantic_score > 0 else
                                "emotional" if r.emotional_score > 0 else
                                "spreading" if r.spreading_score > 0 else
                                "involuntary"),
                    "emotion": r.memory.emotion.to_dict(),
                }
                for r in retrieved
            ],
        })

    scored = score_agent("eaam", results)

    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2, default=str)
    print(f"[EAAM] Done: overall={scored['overall']:.4f}")
    return scored


# ============================================================================
# COMPARISON REPORT
# ============================================================================

def compare_results(vanilla_path: str, eaam_path: str):
    """Load both results and print comparison."""
    with open(vanilla_path) as f:
        vanilla = json.load(f)
    with open(eaam_path) as f:
        eaam = json.load(f)

    print("\n" + "=" * 75)
    print("DIALOG BENCHMARK: Vanilla AI vs AI + EAAM Memory")
    print("=" * 75)

    print(f"\n  OVERALL:")
    print(f"    Vanilla AI:    {vanilla['overall']:.4f}")
    print(f"    AI + EAAM:     {eaam['overall']:.4f}")
    delta = eaam['overall'] - vanilla['overall']
    if vanilla['overall'] > 0:
        pct = (delta / vanilla['overall']) * 100
    else:
        pct = float('inf') if delta > 0 else 0
    print(f"    Delta:         {delta:+.4f} ({pct:+.1f}%)")
    print(f"    Winner:        {'AI + EAAM' if delta > 0 else 'Vanilla AI'}")

    print(f"\n  BY CATEGORY:")
    all_cats = sorted(set(list(vanilla.get('categories', {}).keys()) +
                          list(eaam.get('categories', {}).keys())))
    for cat in all_cats:
        v = vanilla.get('categories', {}).get(cat, 0)
        e = eaam.get('categories', {}).get(cat, 0)
        d = e - v
        w = "EAAM" if d > 0.01 else ("TIE" if abs(d) <= 0.01 else "Vanilla")
        print(f"    {cat:25} Vanilla={v:.3f}  EAAM={e:.3f}  Δ={d:+.3f}  {w}")

    print(f"\n  PER-QUERY BREAKDOWN:")
    for vq, eq in zip(vanilla.get('per_query', []), eaam.get('per_query', [])):
        query = next(q for q in TEST_QUERIES if q.id == vq['query_id'])
        vc = vq.get('composite', 0)
        ec = eq.get('composite', 0)
        sessions_found = eq.get('retrieved_sessions', [])
        expected = query.expected_sessions
        matched = set(sessions_found) & set(expected)
        print(f"\n    [{query.category}] {query.id}")
        print(f"      Query: \"{query.query[:65]}...\"")
        print(f"      Vanilla: {vc:.3f}  |  EAAM: {ec:.3f}  |  Δ={ec-vc:+.3f}")
        print(f"      Sessions expected: {expected}")
        print(f"      Sessions found:    {list(matched)} ({len(matched)}/{len(expected)})")
        kw_hits = sum(1 for kw in query.expected_keywords
                      if kw.lower() in " ".join(eq.get('retrieved_texts', [])).lower())
        print(f"      Keywords hit: {kw_hits}/{len(query.expected_keywords)} {query.expected_keywords[:4]}...")

        # Show what EAAM actually retrieved
        if "retrieval_details" in eq:
            print(f"      EAAM retrieved:")
            for rd in eq.get("retrieval_details", [])[:3]:
                print(f"        [{rd['pathway']:9}] [{rd['session']:20}] {rd['content'][:55]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["vanilla", "eaam", "compare"], required=True)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--vanilla-path", type=str, default=str(_ROOT / "results" / "dialog" / "vanilla.json"))
    parser.add_argument("--eaam-path", type=str, default=str(_ROOT / "results" / "dialog" / "eaam.json"))
    args = parser.parse_args()

    os.makedirs(_ROOT / "results" / "dialog", exist_ok=True)

    if args.agent == "vanilla":
        out = args.output or args.vanilla_path
        run_vanilla_agent(out)
    elif args.agent == "eaam":
        out = args.output or args.eaam_path
        run_eaam_agent(out)
    elif args.agent == "compare":
        compare_results(args.vanilla_path, args.eaam_path)
