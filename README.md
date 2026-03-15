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

## MemBlock vs Mem0 vs Traditional Memory — An Honest Comparison

We respect what [Mem0](https://mem0.ai) has built — they're the most adopted memory layer in the ecosystem with 41k+ GitHub stars and proven benchmarks. This comparison is honest: we call out where Mem0 wins, where MemBlock wins, and where traditional approaches still make sense.

### Head-to-Head: MemBlock vs Mem0

| Capability | **Mem0** | **MemBlock** | Verdict |
|---|---|---|---|
| **Typed memory structure** | Unstructured — memories are text blobs with metadata | **5 built-in types** (Fact, Preference, Event, Entity, Relation) | **MemBlock** — structure matters for reasoning |
| **Knowledge graph** | Available on Pro tier ($249/mo) | **Built-in, no paywall** — 8 relationship types with traversal | **MemBlock** — graph shouldn't be a premium feature |
| **Memory decay** | TTL-based expiration | **Exponential decay + access reinforcement** — mimics how human memory works | **MemBlock** — smarter forgetting |
| **Deduplication** | Basic duplicate detection | **Exact hash + semantic similarity** with 4 configurable policies (error, skip, return, merge) | **MemBlock** — more control |
| **Encryption** | Platform-managed (cloud) | **AES-256-GCM, field-level, your keys, your control** | **MemBlock** — zero-trust encryption |
| **Tamper detection** | Audit logs (enterprise) | **SHA-256 hash chain on every operation** — cryptographic proof | **MemBlock** — verifiable integrity |
| **Search** | Vector + graph (Pro) | **FTS + vector + RRF fusion** | Tie — different approaches, both effective |
| **Ecosystem & integrations** | Python, JS, LangGraph, CrewAI, Vercel AI SDK, MCP server | Python only | **Mem0** — much broader ecosystem |
| **Community & adoption** | 41k+ GitHub stars, 186M+ API calls | New, private distribution | **Mem0** — battle-tested at scale |
| **Managed cloud option** | Yes — zero-infra hosted service | No — self-hosted only | **Mem0** — if you don't want to manage infrastructure |
| **Token compression** | Up to 80% prompt token reduction (claimed) | Context builder with token budgets | **Mem0** — more mature optimization |
| **Framework integrations** | OpenAI, LangChain, CrewAI, Vercel AI SDK | Standalone Python SDK | **Mem0** — plug into any stack |
| **Data ownership** | Cloud: vendor-hosted. Self-hosted: you own it | **100% your infrastructure, always** | **MemBlock** — no cloud option means no data risk |
| **Cost** | Free tier → $19/mo → $249/mo (graph requires Pro) | **One-time license, no subscriptions** | **MemBlock** — no recurring cost |
| **Setup complexity** | Moderate (API keys, config, embeddings) | **One line:** `MemBlock(storage="sqlite:///db")` | **MemBlock** — simpler to start |
| **LLM auto-extraction** | Automatic from conversations | **Built-in (OpenAI, Anthropic)** with configurable triggers | Tie — both handle this well |
| **Multi-tenancy** | User/session/agent levels | **User + session isolation** (PostgreSQL) | Tie — both support user and session scoping |

### When to Choose What

**Choose Mem0 if:**
- You need a managed cloud service and don't want to host anything
- You're building in JavaScript/TypeScript (MemBlock is Python only)
- You need integrations with LangGraph, CrewAI, or Vercel AI SDK out of the box
- You need enterprise compliance (SOC 2) and don't want to handle it yourself
- Your team values community size and ecosystem maturity over raw feature set

**Choose MemBlock if:**
- You need full data ownership with zero cloud dependencies
- You want typed memory structure, knowledge graph, and decay without paying $249/month
- You need field-level encryption with your own keys
- You need cryptographic tamper detection (hash chain verification)
- You want a simple, deterministic SDK you can test and debug like any other library
- You're cost-sensitive — one license, no recurring subscription

**Choose traditional approaches (vector DB, JSON files) if:**
- You only need similarity search and nothing else (use a vector DB directly)
- Your project is a quick prototype that doesn't need persistence (just use a dict)
- You're comfortable building and maintaining custom memory infrastructure

### The Honest Bottom Line

Mem0 is the **market leader** — bigger community, more integrations, managed cloud, proven at scale. If you want the safe, well-supported choice with zero infrastructure work, Mem0 is excellent.

MemBlock is for developers who want **more structure, more control, and more ownership**. Typed memories, built-in graph (no paywall), cryptographic integrity, field-level encryption, and intelligent decay — all running on your own database with no subscriptions. It's a different philosophy: you own everything, you control everything.

We're not trying to replace Mem0. We're building for developers who need what Mem0 doesn't offer.

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

### Session Scoping
Optional `session_id` on every block — scope memories to individual conversations, threads, or workflows. Query within a session, list all sessions, or retrieve full session history. Fully backwards compatible — existing code works unchanged. Combined with `user_id` (PostgreSQL), you get user + session isolation for multi-tenant, multi-conversation products.

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
