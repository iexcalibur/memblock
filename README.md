<p align="center">
  <h1 align="center">MemBlock</h1>
  <p align="center"><strong>Structured memory SDK for AI agents.</strong></p>
  <p align="center">Give your AI applications persistent, queryable, and intelligent memory — without cloud dependencies.</p>
  <p align="center"><code>Python 3.10+</code> · <code>335 Tests</code> · <code>Private Distribution</code></p>
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
| **Session support** | Three levels: `user_id`, `agent_id`, `run_id` — built-in hierarchy | **Opt-in `session_id`** — single-session by default, multi-session when you need it | **Mem0** — more granular hierarchy with agent-level scoping |
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
- **Cost** — One license, no subscriptions. Graph, encryption, decay, sessions — all included.

### When to Choose What

**Choose Mem0 if:**
- You need a managed cloud service and don't want to host anything
- You're building in JavaScript/TypeScript or need REST API access
- You need integrations with LangChain, CrewAI, AutoGen, or Vercel AI SDK
- You need enterprise compliance (SOC 2, HIPAA) handled for you
- You value ecosystem maturity and community support
- Your graph memory needs are complex enough to justify $249/mo

**Choose MemBlock if:**
- You need full data ownership with zero cloud dependencies
- You want typed memory, knowledge graph, and decay without a $249/mo paywall
- You need field-level encryption with your own keys at every tier
- You need cryptographic tamper detection (not just audit logs)
- You want a simple, deterministic SDK you can test and debug like any other library
- You're building in Python and want minimal infrastructure (SQLite is enough to start)
- You're cost-sensitive — one license, everything included

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
pip install https://github.com/iexcalibur/memblock/releases/download/v0.3.0/memblock-0.3.0-py3-none-any.whl
```

### Optional Extras

```bash
# PostgreSQL support
pip install "memblock[postgres]"

# Local embeddings (CPU, no API key needed)
pip install "memblock[embeddings]"

# LLM extraction (OpenAI + Anthropic)
pip install "memblock[llm]"

# Everything
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
├── SessionScoping  — Optional session_id isolation per block
└── StorageAdapter  — SQLite or PostgreSQL (swappable)
```

Every component is testable in isolation. The facade composes them into a single clean API.

---

## License & Ownership

**Copyright (c) 2025-2026 iexcalibur. All Rights Reserved.**

This software is **proprietary and confidential**. It is not open-source.

- You may **NOT** copy, modify, distribute, sublicense, or sell this software without prior written permission.
- You may **NOT** reverse engineer, decompile, or create derivative works.
- You may **NOT** redistribute in any form — source code or compiled — without explicit authorization.

Access is granted on an invite-only basis to authorized individuals and organizations.

Unauthorized use, reproduction, or distribution of this software is strictly prohibited and may result in legal action.
