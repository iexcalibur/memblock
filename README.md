# MemBlock

**Structured memory SDK for AI agents.**

Give your AI applications persistent, queryable, and intelligent memory — without cloud dependencies.

`Python 3.10+` · `319 Tests` · `Private Distribution`

---

## What is MemBlock?

Every AI agent has the same problem: **it forgets.**

Between sessions, between conversations, between deployments — context disappears. Most solutions either lock you into a cloud vendor, dump everything into an unstructured vector database, or force you to build memory infrastructure from scratch.

**MemBlock is different.** It's a Python SDK that gives your AI agents structured, typed, graph-connected memory that lives in your own database. You control what gets remembered, how it's organized, when it decays, and who can access it.

No cloud subscriptions. No vendor lock-in. No "trust us with your data." Just a clean Python API backed by SQLite or PostgreSQL.

---

## Why MemBlock?

### The problem with current approaches

| Pain Point | What usually happens |
|---|---|
| **Context loss** | Agents restart with zero knowledge every session |
| **Unstructured dumps** | Raw vector DBs can't distinguish a fact from a preference |
| **Cloud lock-in** | Memory-as-a-service means your data lives on someone else's servers |
| **No forgetting** | Everything is stored forever with equal weight — no relevance decay |
| **No relationships** | Flat storage can't represent "X contradicts Y" or "A is part of B" |
| **Retrieval noise** | Keyword OR vector search alone produces inconsistent results |

### What MemBlock gives you

- **Typed memories** — FACT, PREFERENCE, EVENT, ENTITY, RELATION — not just blobs of text
- **Knowledge graph** — Link memories with typed relationships (supports, contradicts, caused_by, etc.)
- **Intelligent decay** — Memories weaken over time unless reinforced by access
- **Hybrid search** — Combines full-text search + vector similarity for better recall
- **Your database** — SQLite for local dev, PostgreSQL for production. You own the data.
- **Deterministic APIs** — No black boxes. Every operation is testable and predictable.

---

## Who is MemBlock For?

MemBlock is built for **developers who build AI-powered applications**:

- **AI agent developers** building copilots, assistants, or autonomous agents that need persistent context
- **Product teams** shipping LLM features that require reliable memory across user sessions
- **Backend engineers** who want a structured memory layer they can test, debug, and deploy like any other component
- **Teams with data control requirements** — regulated industries, enterprise, or anyone who can't send user data to a third-party memory service

If you're building with OpenAI, Anthropic, or any LLM provider and need your agent to actually remember things — MemBlock is for you.

---

## How MemBlock Compares

| Capability | Raw Vector DB | Cloud Memory API | JSON Files | **MemBlock** |
|---|---|---|---|---|
| **Structured types** | No | Varies | Manual | **5 built-in types** |
| **Graph relationships** | No | Rarely | No | **8 relation types** |
| **Memory decay** | No | No | No | **Exponential + access boost** |
| **Deduplication** | Manual | Varies | Manual | **Exact + semantic** |
| **Encryption** | Varies | Provider-controlled | No | **AES-256-GCM, field-level** |
| **Hybrid search** | Vector only | Varies | None | **FTS + vector + RRF merge** |
| **Data ownership** | Self-hosted possible | Vendor-owned | Local | **Fully local** |
| **Tamper detection** | No | No | No | **SHA-256 hash chain** |
| **LLM extraction** | No | Sometimes | No | **Built-in (OpenAI, Anthropic)** |
| **Cost** | Infra + embedding APIs | Subscription | Free | **Free (self-hosted)** |

---

## Features

### Typed Block Storage
Store memories as structured blocks with confidence scores, source tracking, tags, and per-block encryption. Five built-in types: **FACT**, **PREFERENCE**, **EVENT**, **ENTITY**, **RELATION**.

### Knowledge Graph
Connect memories with typed edges: `supports`, `contradicts`, `caused_by`, `related_to`, `part_of`, `derived_from`, `supersedes`, `about`. Traverse relationships, detect contradictions, and discover clusters.

### Hybrid Search
Full-text search (FTS5) combined with vector similarity search, merged using Reciprocal Rank Fusion. Better retrieval than either approach alone. Supports local embeddings (FastEmbed), OpenAI, and Gemini providers.

### Memory Decay Engine
Memories naturally weaken over time using exponential decay. Frequently accessed memories stay strong. Configurable decay rates per block. Auto-prune weak memories below a threshold. TTL support for time-limited memories.

### AES-256 Encryption
Field-level encryption with AES-256-GCM. PBKDF2-SHA256 key derivation with 480,000 iterations. Encrypt content only (STANDARD) or content + tags (SENSITIVE). Passphrase-based — no key management infrastructure required.

### Smart Deduplication
Four policies: `error`, `skip`, `return_existing`, `merge`. Two detection layers: exact match (SHA-256 content hash) and semantic similarity (cosine threshold). Catches both identical and near-identical memories.

### LLM Auto-Extraction
Extract structured memories from raw conversations using OpenAI or Anthropic models. Buffer messages and trigger extraction at configurable intervals. Automatic type detection, confidence scoring, and relationship linking.

### Context Builder
Build LLM-ready context strings from relevant memories. Three strategies: **relevance** (best matches first), **graph_walk** (follow relationships), **type_grouped** (organized by memory type). Token budget enforcement included.

### Tamper Detection
Append-only operation log with SHA-256 hash chain. Every create, update, delete, link, and unlink operation is recorded. Verify integrity at any time with a single method call.

### Multi-Storage Support
**SQLite** for local development and single-user applications. **PostgreSQL** for production, multi-tenant deployments with user-level isolation. Same API, swap the connection string.

### CLI Tools
`memblock init` · `memblock query` · `memblock stats` · `memblock prune` · `memblock export` · `memblock reindex` · `memblock version`

---

## Quick Start

```python
from memblock import MemBlock, BlockType

# Initialize with a local SQLite database
mem = MemBlock(storage="sqlite:///./agent_memory.db")

# Store structured memories
mem.store("User prefers Python over JavaScript", type=BlockType.PREFERENCE)
mem.store("User works at Acme Corp", type=BlockType.FACT, confidence=0.95)
event = mem.store("Deployed v2.0 on March 10", type=BlockType.EVENT)

# Query with hybrid search
results = mem.query(text_search="programming language", type=BlockType.PREFERENCE)

# Build LLM-ready context
context = mem.build_context(query="user preferences", token_budget=4000)

# Link related memories
mem.link(results[0].id, event.id, relation="related_to")

# Check integrity
report = mem.verify()
print(f"Tamper detected: {report.tampered}")

mem.close()
```

---

## Installation

MemBlock is distributed privately. Contact the author for access.

### From GitHub Release (collaborators)

```bash
pip install https://github.com/iexcalibur/memblock/releases/download/v0.2.0/memblock-0.2.0-py3-none-any.whl
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

## API Overview

| Category | Methods |
|---|---|
| **Store** | `store()`, `get()`, `update()`, `delete()` |
| **Graph** | `link()`, `unlink()`, `neighbors()`, `traverse()` |
| **Search** | `query()`, `build_context()` |
| **Extract** | `extract()`, `extract_messages()`, `add_message()`, `flush_extraction()` |
| **Manage** | `prune()`, `strongest()`, `weakest()`, `verify()`, `stats()`, `export_markdown()` |

---

## License & Ownership

Copyright (c) 2025-2026 **iexcalibur** (Shubham Kannojia). All Rights Reserved.

This software is **proprietary and confidential**. Unauthorized copying, modification, distribution, or use of this software, via any medium, is strictly prohibited without prior written permission from the copyright holder.

This is not open-source software. Access is granted on an invite-only basis.

For licensing inquiries: **shubhamkannojia10@gmail.com**
