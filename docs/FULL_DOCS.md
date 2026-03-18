<p align="center">
  <h1 align="center">MemBlock</h1>
  <p align="center"><strong>Structured memory SDK for AI agents.</strong></p>
  <p align="center">Give your AI applications persistent, queryable, and intelligent memory — without cloud dependencies.</p>
  <p align="center"><code>Python 3.10+</code> · <code>510 Tests</code> · <code>v0.6.1</code> · <code>Private Distribution</code></p>
</p>

---

## The Problem

Every AI agent has the same problem: **it forgets.**

Between sessions, between conversations, between deployments — context disappears. The solutions available today are broken in different ways:

- **Vector databases** give you similarity search but nothing else. They can't tell a fact from a preference. No structure, no relationships, no decay.
- **Cloud memory APIs** lock you into a subscription and store your users' data on someone else's servers. You have zero control over what happens to it.
- **Custom JSON/file solutions** work until they don't. No search, no dedup, no integrity guarantees. You end up rebuilding the same infrastructure every project.
- **Conversation history** isn't memory. Dumping 50k tokens of chat history into a context window is expensive, noisy, and doesn't scale.

There is no developer-first memory SDK that gives you structure, search, relationships, decay, encryption, and data ownership — all in one package, all running on your own infrastructure.

**Until now.**

---

## What is MemBlock?

MemBlock is a Python SDK that gives AI agents **structured, typed, graph-connected memory** backed by your own database.

You control what gets remembered, how it's organized, when it decays, and who can access it. No cloud subscriptions. No vendor lock-in. No "trust us with your data."

```python
from memblock import MemBlock, BlockType

mem = MemBlock(storage="sqlite:///./agent_memory.db")

# Store structured memories — not just text blobs
mem.store("User prefers Python over JavaScript", type=BlockType.PREFERENCE)
mem.store("User works at Acme Corp", type=BlockType.FACT, confidence=0.95)
event = mem.store("Deployed v2.0 on March 10", type=BlockType.EVENT)

# Query with hybrid search (FTS + vector similarity)
results = mem.query(text_search="programming language", type=BlockType.PREFERENCE)

# Build LLM-ready context — drop straight into your prompt
context = mem.build_context(query="user preferences", token_budget=4000)

# Link related memories into a knowledge graph
mem.link(results[0].id, event.id, relation="related_to")

# Verify nothing has been tampered with
report = mem.verify()

mem.close()
```

That's it. Five lines to go from "my agent forgets everything" to "my agent has structured, searchable, graph-connected memory."

---

## MemBlock vs Mem0 — An Honest Comparison (Updated March 2026)

Mem0 is the dominant memory layer in the AI ecosystem — ~50k GitHub stars, $24M Series A, AWS Agent SDK partnership, and 50+ integrations. They've earned their position. This comparison is honest: we call out where Mem0 wins, where MemBlock wins, and where it's a tie.

### Head-to-Head

| Capability | **Mem0** | **MemBlock** | Verdict |
|---|---|---|---|
| **Typed memory structure** | Unstructured — memories are text blobs with metadata | **5 built-in types** (Fact, Preference, Event, Entity, Relation) with per-type behavior | **MemBlock** — structure enables smarter retrieval and reasoning |
| **Knowledge graph** | Neo4j-based graph memory — **Pro tier only ($249/mo)**. LLM extracts entities/relations automatically | **Built-in, no paywall** — 8 typed relationship types with traversal. Manual or auto-extraction | **Depends** — Mem0's auto-extraction is more sophisticated; MemBlock's is free and built-in |
| **Memory decay** | Memories adapt over time as preferences change. Conflict resolution via update resolver | **Exponential decay + access reinforcement** — configurable per-block. Auto-prune weak memories. TTL support | **MemBlock** — explicit, deterministic decay you can test and tune |
| **Deduplication** | Automatic duplicate detection and conflict resolution | **Exact hash + semantic similarity** with 4 configurable policies (error, skip, return, merge) | **MemBlock** — more developer control over dedup behavior |
| **Encryption** | SOC 2, HIPAA compliant. BYOK on enterprise. Platform-managed | **AES-256-GCM, field-level, your keys, always** — STANDARD (content) or SENSITIVE (content + tags) | **MemBlock** — zero-trust encryption without enterprise tier |
| **Tamper detection** | Audit trails, versioned + timestamped memories. Exportable | **SHA-256 hash chain on every operation** — cryptographic proof of integrity, verifiable in one call | **MemBlock** — cryptographic vs log-based verification |
| **Search** | Semantic retrieval via embeddings. 22+ vector store backends. Filtering + batch ops | **FTS5 + vector + RRF fusion** — hybrid search combining keyword and semantic | Tie — different strengths. Mem0 has more backend options; MemBlock has hybrid fusion |
| **Session support** | Three levels: `user_id`, `agent_id`, `run_id` — built-in hierarchy | **5-level hierarchy**: `org_id` → `project_id` → `user_id` → `agent_id` → `session_id` — all opt-in | **MemBlock** — deeper hierarchy with org/project scoping |
| **Multi-tenancy** | Automatic graph isolation per user. Query time constant as users scale | **User + session isolation** (PostgreSQL). Composite indexes for fast scoped queries | Tie — both isolate by user. Mem0 scales graph isolation; MemBlock scales relational queries |
| **Ecosystem** | Python, JS/TS, REST API. LangChain, LangGraph, CrewAI, AutoGen, Vercel AI SDK, MCP server, Chrome extension. AWS Agent SDK partner | Python only. Standalone SDK | **Mem0** — significantly broader. Not close |
| **Community** | ~50k GitHub stars. $24M Series A. 18M+ PyPI downloads | New, private distribution | **Mem0** — massive community and proven at scale |
| **Cloud option** | Managed platform at api.mem0.ai. Free tier (10K memories) → Pro → Enterprise | No cloud — self-hosted only | **Mem0** — if you want zero infrastructure |
| **Self-hosted** | Docker Compose (3 containers: API + PostgreSQL/pgvector + Neo4j). Apache 2.0 | `pip install` + SQLite (zero infra) or PostgreSQL | **MemBlock** — simpler self-hosting. No Neo4j or Docker required |
| **Token optimization** | Claims 91% faster responses, 90% fewer tokens vs full history | Context builder with token budgets and 3 strategies (relevance, graph walk, type grouped) | **Mem0** — more battle-tested optimization at scale |
| **Data ownership** | Cloud: vendor-hosted. Self-hosted: you own it (Apache 2.0) | **100% your infrastructure, always**. No cloud option means no data risk | Tie — both offer self-hosted. MemBlock has no cloud temptation |
| **Cost** | Free (10K) → $19/mo (50K) → $249/mo (unlimited + graph) → Enterprise | **One-time license, no subscriptions** | **MemBlock** — no recurring cost. Graph included free |
| **Setup** | Cloud: API key + 3 lines. Self-hosted: Docker Compose + config | **One line:** `MemBlock(storage="sqlite:///db")` | **MemBlock** — simpler to start. Mem0 cloud is also easy though |
| **LLM extraction** | Automatic from conversations. Supports 15+ LLM providers | **Built-in (OpenAI, Anthropic)** with configurable triggers and buffer-based extraction | Tie — both handle this. Mem0 supports more providers |
| **Async API** | Async-by-default since v1.0 | **AsyncMemBlock** — full async wrapper with `asyncio.to_thread()` | Tie — both support async |
| **Reranking** | Cohere, HuggingFace, BM25 rerankers | **BM25 (zero deps), Cohere, CrossEncoder** — pluggable reranker interface | Tie — similar offerings |
| **Conflict resolution** | LLM-powered ADD/UPDATE/DELETE decisions on every `add()` | **Opt-in LLM conflict resolution** — same ADD/UPDATE/DELETE/NONE decisions, configurable | Tie — both offer this. Mem0's is on by default |
| **Event hooks** | Not available | **on_add, on_update, on_delete, on_query** — sync + async callbacks | **MemBlock** — Mem0 doesn't have this |
| **Metadata filtering** | Arbitrary key-value filters on search | **Custom metadata JSON** with arbitrary key-value filtering on SQLite + PostgreSQL | Tie — both support this |
| **Auto-extraction on store** | Automatic on every `add()` by default | **Opt-in `auto_extract_on_store`** — extracted blocks linked via DERIVED_FROM | Tie — both offer this. Different defaults |
| **Offline / air-gapped** | Self-hosted with Ollama — fully offline possible | **SQLite + local FastEmbed** — fully offline, no Docker | **MemBlock** — simpler offline story |

### What Mem0 Does Better

Let's be direct about where Mem0 wins outright:

- **Ecosystem reach** — Python, JS/TS, REST API, Chrome extension, MCP server, AWS partnership. MemBlock is Python only.
- **Graph intelligence** — Their Neo4j-based graph with LLM-powered entity extraction and conflict resolution is more sophisticated than our manual graph.
- **Scale proof** — 50k stars, $24M funding, enterprise customers. MemBlock is new and unproven at scale.
- **Zero-config cloud** — Sign up, get an API key, start storing memories. No infrastructure to think about.
- **Framework integrations** — Drop-in support for LangChain, CrewAI, AutoGen, and more. MemBlock requires manual integration.

### What MemBlock Does Better

- **Typed structure** — 5 memory types with per-type behavior vs text blobs. Your agent knows the difference between a fact and a preference.
- **Free knowledge graph** — Built-in at every tier. Mem0 paywalls graph behind $249/mo.
- **Deterministic decay** — Exponential decay with access reinforcement, configurable per block. Testable and predictable.
- **Field-level encryption** — AES-256-GCM with your keys, no enterprise tier required. Encrypt content only or content + tags.
- **Cryptographic integrity** — SHA-256 hash chain on every operation. One call to verify nothing was tampered with.
- **Setup simplicity** — `pip install` + one line of Python. No Docker, no Neo4j, no API keys needed for basic use.
- **Event hooks** — Register callbacks on memory lifecycle events (add, update, delete, query). Mem0 doesn't offer this.
- **Hierarchical scoping** — 5-level opt-in hierarchy (org → project → user → agent → session) vs Mem0's 3 levels.
- **Cost** — One license, no subscriptions. Graph, encryption, decay, sessions — all included.

### Who should choose MemBlock vs Mem0

Use this rule of thumb:

**Choose MemBlock if your priority is:**
- Own data ownership (no memory data sent to external providers by default)
- Structured memory with typed blocks + built-in graph + decay in one package
- Private deployment with SQLite/PostgreSQL on your own infrastructure
- Low recurring cost and private distribution control
- Deterministic, debuggable behavior under your own CI/security policies
- Full control over encryption, retention, and schema evolution

**Choose Mem0 if your priority is:**
- Fast managed rollout with less backend operations burden
- Broad language and orchestration ecosystem integration speed
- Centralized cloud service model with external API-based scale
- Team preference for API-first usage over local SDK ownership
- Faster time-to-production for multi-stack teams

**Decision matrix:**

| Requirement | MemBlock | Mem0 |
|---|---|---|
| Data cannot leave your infra | **Strong fit** | Managed with external dependency |
| Need private on-prem or VPC style deployments | **Strong fit** | Conditional |
| Need a mature multi-language integration story | Possible (Python-first) | **Strong fit** |
| Need local-first + source-level control | **Strong fit** | Limited |
| Want minimal backend operational responsibility | Possible but you operate it | **Strong fit** |
| Budget prefers one-time/proprietary licensing model | **Strong fit** | API usage cost model |

**Quick rule:**  
Use **MemBlock** when control and privacy are non-negotiable.  
Use **Mem0** when managed operations and broad ecosystem plug-and-play matter more.

### The Honest Bottom Line

Mem0 is the **market leader** — bigger community, more integrations, managed cloud, $24M in funding, and proven at enterprise scale. If you want the safe, well-supported choice with zero infrastructure work, Mem0 is excellent. Their self-hosted option (Apache 2.0) is also solid.

MemBlock is for developers who want **more structure, more control, and a simpler stack**. Typed memories, free knowledge graph, cryptographic integrity, field-level encryption, intelligent decay, and session scoping — all running on SQLite or PostgreSQL with no Docker, no Neo4j, and no subscriptions. It's a different philosophy: everything included, everything you control, everything testable.

We're not trying to replace Mem0. We're building for developers who need what Mem0 either paywalls or doesn't offer.

---

## Who is MemBlock For?

MemBlock is built with a **developer-first mindset**. No GUIs, no dashboards, no drag-and-drop. Just a clean Python API that works the way you expect.

### AI Agent Developers
Building copilots, assistants, or autonomous agents that need to remember context across sessions. MemBlock gives your agent real memory — not just a bigger context window.

### Product Teams Shipping LLM Features
Need reliable memory across user sessions for your product? MemBlock gives you structured storage, deterministic APIs, and predictable behavior. No black boxes.

### Backend Engineers
You want a memory layer you can test, debug, monitor, and deploy like any other backend component. MemBlock is a library, not a service. It runs in your process, uses your database.

### Teams With Data Control Requirements
Regulated industries, enterprise, healthcare, finance — anyone who can't send user data to a third-party memory service. MemBlock runs entirely on your infrastructure. Your data never leaves your systems.

### Solo Builders & Indie Hackers
Building an AI product and don't want to pay $50/month for a memory API? MemBlock is free to use once licensed. SQLite is all you need to get started.

---

## Core Features

### Typed Block Storage
Store memories as structured blocks — not just raw text. Each block has a **type** (FACT, PREFERENCE, EVENT, ENTITY, RELATION), a **confidence score**, **source tracking**, **tags**, and optional **per-block encryption**. You always know what kind of memory you're dealing with.

### Knowledge Graph
Connect memories with typed edges: `supports`, `contradicts`, `caused_by`, `related_to`, `part_of`, `derived_from`, `supersedes`, `about`. Traverse relationships, detect contradictions, and discover context clusters. Your agent doesn't just remember facts — it understands how they connect.

### Hybrid Search
Full-text search combined with vector similarity, merged using Reciprocal Rank Fusion (RRF). Better retrieval than either approach alone. Supports local embeddings (no API key needed), OpenAI, and Gemini providers.

### Memory Decay Engine
Memories naturally weaken over time using exponential decay. Frequently accessed memories stay strong — just like human memory. Configurable decay rates per block. Auto-prune weak memories. TTL support for time-limited memories.

### AES-256 Encryption
Field-level encryption with AES-256-GCM. Encrypt content only (STANDARD) or content + tags (SENSITIVE). Passphrase-based key derivation — no key management infrastructure required.

### Smart Deduplication
Four policies: `error`, `skip`, `return_existing`, `merge`. Two detection layers: exact content hash and semantic similarity (cosine threshold). Catches both identical and near-identical memories before they pollute your store.

### LLM Auto-Extraction
Extract structured memories from raw conversations using OpenAI or Anthropic models. Buffer messages and trigger extraction at configurable intervals. Automatic type detection, confidence scoring, and relationship linking — turns conversations into structured knowledge.

### Context Builder
Build LLM-ready context strings from relevant memories. Three strategies: **relevance** (best matches), **graph_walk** (follow relationships), **type_grouped** (organized by type). Token budget enforcement ensures you never exceed your context window.

### Tamper Detection
Append-only operation log with SHA-256 hash chain. Every create, update, delete, link, and unlink is recorded. Verify integrity at any time — one method call tells you if anything has been modified outside the SDK.

### Session Scoping (Opt-in)
Multi-session support is **off by default** — all memories live in a single global scope, which is all most agents need. When you're ready for multi-conversation products, just start passing `session_id` to any method. No config changes, no migrations, no feature flags — it just works.

```python
# Single-session (default) — no session_id needed
mem.store("User likes Python", type=BlockType.PREFERENCE)
mem.query(type=BlockType.PREFERENCE)  # returns all blocks

# Multi-session (opt-in) — pass session_id when you need it
mem.store("msg in chat 1", type=BlockType.EVENT, session_id="chat_001")
mem.store("msg in chat 2", type=BlockType.EVENT, session_id="chat_002")
mem.query(session_id="chat_001")  # only chat_001 blocks
mem.get_sessions()                # ["chat_001", "chat_002"]
mem.get_session_history("chat_001")  # chronological blocks
```

The developer decides. Combined with `user_id` (PostgreSQL), you get user + session isolation for multi-tenant, multi-conversation products.

### Async API
Full async support via `AsyncMemBlock`. Every method is wrapped with `asyncio.to_thread()` so it won't block your event loop. Works with FastAPI, aiohttp, and any async framework.

```python
from memblock import AsyncMemBlock, BlockType

async with AsyncMemBlock(storage="sqlite:///./memory.db") as mem:
    block = await mem.store("User prefers Python", type=BlockType.PREFERENCE)
    results = await mem.query(text_search="Python")
    context = await mem.build_context(query="preferences", token_budget=4000)
```

### Reranker Support
Improve search quality with pluggable rerankers. BM25 (zero dependencies), Cohere (API), or CrossEncoder (local HuggingFace model). Reranking happens automatically after the initial search.

```python
from memblock.rerankers import BM25Reranker, CohereReranker

# BM25 — zero dependencies, works offline
mem = MemBlock(storage="sqlite:///./db", reranker=BM25Reranker())

# Cohere — best quality
mem = MemBlock(storage="sqlite:///./db", reranker=CohereReranker(api_key="co-..."))
```

### Conflict Resolution via LLM
When enabled, `store()` finds semantically similar existing memories and asks the LLM to decide: **ADD** (new), **UPDATE** (refine existing), **DELETE** (contradicted), or **NONE** (already known). Same concept as Mem0's inference mode.

```python
mem = MemBlock(
    storage="sqlite:///./db",
    embeddings=True,
    extract_provider="openai",
    extract_api_key="sk-...",
    conflict_resolution=True,  # enable LLM conflict resolution
)
```

### Hierarchical Scoping
5-level opt-in hierarchy: `org_id` → `project_id` → `user_id` → `agent_id` → `session_id`. Set defaults in the constructor, override per-call. Fully backward compatible — existing code works unchanged.

```python
mem = MemBlock(
    storage="postgresql://...",
    org_id="acme_corp",
    project_id="chatbot_v2",
    user_id="u_123",
    agent_id="support_bot",
)
mem.store("User prefers email", type=BlockType.PREFERENCE, session_id="chat_001")
mem.query(org_id="acme_corp", project_id="chatbot_v2")  # scoped query
```

### Custom Metadata Filtering
Attach arbitrary key-value metadata to any block and filter on it during search.

```python
mem.store("User has Pro plan", type=BlockType.FACT, metadata={"plan": "pro", "region": "us-east"})
results = mem.query(metadata_filters={"plan": "pro"})
```

### Event Hooks
Register callbacks for memory lifecycle events. Sync or async. Errors in hooks never break the main operation.

```python
def on_memory_added(data):
    print(f"New memory: {data['block_id']}")

mem = MemBlock(storage="sqlite:///./db")
mem.on("on_add", on_memory_added)
mem.on("on_delete", lambda data: log.warning(f"Deleted: {data['block_id']}"))
```

### Connection Pooling (PostgreSQL)
Production deployments need connection pooling. Pass a `psycopg_pool.ConnectionPool` to avoid per-request connection overhead. Connections are borrowed from the pool and returned on close — no changes to your application code.

```python
from psycopg_pool import ConnectionPool
from memblock import MemBlock

pool = ConnectionPool("postgresql://user:pass@localhost/mydb", min_size=4, max_size=20)
mem = MemBlock(storage="postgresql://user:pass@localhost/mydb", user_id="u_123", pool=pool)
# connections are borrowed from the pool, returned on mem.close()
```

### MemBlockPool (Instance Factory)
Manage per-user MemBlock instances with LRU caching. Ideal for multi-tenant FastAPI/Django backends serving many concurrent users without creating unbounded connections.

```python
from memblock import MemBlockPool, AsyncMemBlockPool

# Sync
with MemBlockPool("postgresql://...", max_instances=100) as pool:
    mem = pool.get("user_123")  # cached or created
    mem.store("Hello", type=BlockType.FACT)

# Async (FastAPI)
async with AsyncMemBlockPool("postgresql://...", max_instances=100) as pool:
    mem = await pool.get("user_456")
    await mem.store("Hello", type=BlockType.FACT)
```

### pgvector Support
When the `pgvector` PostgreSQL extension is installed, MemBlock automatically detects it and uses server-side HNSW vector search instead of brute-force Python cosine similarity. Dual-write keeps backward compatibility — BYTEA embeddings are always stored alongside the pgvector column.

```bash
pip install "memblock[pgvector]"
```

No code changes needed. If pgvector is available in your database, `_hybrid_search` uses `<=>` cosine distance for fast ANN queries. If not, it falls back to Python-side cosine.

### Multi-Storage
**SQLite** for local development and single-user apps. **PostgreSQL** for production multi-tenant deployments with user-level and session-level isolation. Same API — just swap the connection string.

### CLI
```
memblock init       # Initialize a new database
memblock query      # Search memories from the terminal
memblock stats      # Database statistics
memblock prune      # Remove decayed memories
memblock export     # Export to markdown
memblock reindex    # Rebuild search indices
memblock version    # Show version
```

---

## Installation

MemBlock is distributed privately. Access is granted on an invite-only basis.

### From GitHub Release (authorized users)

```bash
pip install https://github.com/iexcalibur/memblock/releases/download/v0.6.1/memblock-0.6.1-py3-none-any.whl
```

### Optional Extras

```bash
# PostgreSQL support
pip install "memblock[postgres]"

# Local embeddings (CPU, no API key needed)
pip install "memblock[embeddings]"

# LLM extraction (OpenAI + Anthropic)
pip install "memblock[llm]"

# Cohere reranker
pip install "memblock[reranker-cohere]"

# Cross-encoder reranker (HuggingFace)
pip install "memblock[reranker-cross-encoder]"

# Connection pooling (psycopg_pool)
pip install "memblock[pool]"

# pgvector support (server-side vector search)
pip install "memblock[pgvector]"

# Everything (cloud-only — works on Python 3.13+)
pip install "memblock[all-cloud]"

# Everything including local embeddings (requires Python ≤ 3.12)
pip install "memblock[all]"
```

---

## API at a Glance

| Category | Methods |
|---|---|
| **Store** | `store()`, `get()`, `update()`, `delete()` |
| **Graph** | `link()`, `unlink()`, `neighbors()`, `traverse()` |
| **Search** | `query()`, `build_context()` |
| **Extract** | `extract()`, `extract_messages()`, `add_message()`, `flush_extraction()` |
| **Sessions** | `get_sessions()`, `get_session_history()` |
| **Hooks** | `on()` — register callbacks for `on_add`, `on_update`, `on_delete`, `on_query` |
| **Async** | `AsyncMemBlock` — full async wrapper for all methods above |
| **Pool** | `MemBlockPool.get()`, `MemBlockPool.remove()`, `MemBlockPool.close_all()` |
| **Manage** | `prune()`, `strongest()`, `weakest()`, `verify()`, `stats()`, `export_markdown()` |

---

## Architecture

MemBlock follows a **composable architecture** — each capability is an independent module composed through a single facade class:

```
MemBlock (facade)
├── BlockStore      — CRUD operations, content hashing, op logging
├── GraphIndex      — Edge management, traversal, neighbor queries
├── QueryEngine     — FTS + vector hybrid search with RRF merge
├── ContextBuilder  — Token-budgeted context generation
├── DecayEngine     — Time-based strength calculation and pruning
├── DuplicateChecker — Exact + semantic dedup
├── CryptoLayer     — AES-256-GCM field-level encryption
├── OpLog           — Append-only hash chain for tamper detection
├── HookManager     — Event callbacks (on_add, on_update, on_delete, on_query)
├── ConflictResolver — LLM-powered ADD/UPDATE/DELETE/NONE decisions
├── Reranker        — BM25, Cohere, CrossEncoder (pluggable)
├── HierarchicalScoping — org → project → user → agent → session
├── MemBlockPool    — LRU-cached per-user instance factory
└── StorageAdapter  — SQLite or PostgreSQL (swappable, pooling + pgvector)
```

Every component is testable in isolation. The facade composes them into a single clean API.

---

## Roadmap

Features planned for future releases:

- **Multi-language SDKs** — TypeScript and Go clients
- **REST API server mode** — Run MemBlock as a standalone API server
- **Multimodal memory** — Store and query images, audio, and documents alongside text
- **Framework integrations** — LangChain, CrewAI, AutoGen, Vercel AI SDK drop-in support
- **MCP server** — Model Context Protocol server for IDE and agent integrations
- **Managed cloud platform** — Optional hosted version for teams that don't want to self-host
- **Additional vector store backends** — Qdrant, Pinecone, ChromaDB support (pgvector shipped in v0.6.0)
- **Token compression** — Smarter context building with LLM-powered summarization
- **Multi-agent memory sharing** — Shared memory pools between agents with access control

Want a feature prioritized? Open an issue or reach out via GitHub.

---

## License & Ownership

**Copyright (c) 2025-2026 iexcalibur. All Rights Reserved.**

This software is **proprietary and confidential**. It is not open-source.

- You may **NOT** copy, modify, distribute, sublicense, or sell this software without prior written permission.
- You may **NOT** reverse engineer, decompile, or create derivative works.
- You may **NOT** redistribute in any form — source code or compiled — without explicit authorization.

Access is granted on an invite-only basis to authorized individuals and organizations.

Unauthorized use, reproduction, or distribution of this software is strictly prohibited and may result in legal action.
