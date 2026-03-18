<p align="center">
  <h1 align="center">MemBlock</h1>
  <p align="center"><strong>Structured memory SDK for AI agents.</strong></p>
  <p align="center">Typed blocks · Knowledge graph · Hybrid search · Encryption · Decay engine — all local, all yours.</p>
  <p align="center"><a href="https://memblock.xyz">memblock.xyz</a></p>
</p>

---

AI agents forget everything between sessions. Vector databases give you search but no structure. Cloud memory APIs lock you in and store your users' data on someone else's servers. MemBlock is the alternative: **typed memory blocks, a built-in knowledge graph, hybrid search, encryption, and intelligent decay** — all running on your infrastructure with `pip install` and one line of Python. No Docker, no Neo4j, no subscriptions. Your data never leaves your machine.

## Install

```bash
pip install memblock
```

## Quick Start

```python
from memblock import MemBlock, BlockType

mem = MemBlock(storage="sqlite:///memory.db")

# Store structured memories
mem.store("User prefers Python", type=BlockType.PREFERENCE)
mem.store("User works at Acme Corp", type=BlockType.FACT, confidence=0.95)

# Query with hybrid search
results = mem.query(text_search="programming", type=BlockType.PREFERENCE)

# Build LLM-ready context
context = mem.build_context(query="user preferences", token_budget=4000)

# Knowledge graph
mem.link(results[0].id, other.id, relation="related_to")

# Tamper detection
mem.verify()
```

## What's Included

- **5 typed memory blocks** — FACT, PREFERENCE, EVENT, ENTITY, RELATION
- **Knowledge graph** — 8 relation types, traversal, no external DB
- **Hybrid search** — FTS5 + vector similarity with Reciprocal Rank Fusion
- **Memory decay** — Exponential decay with access reinforcement
- **AES-256 encryption** — Field-level, your keys, no enterprise tier
- **Tamper detection** — SHA-256 hash chain on every operation
- **LLM extraction** — Auto-extract memories from conversations (OpenAI, Anthropic, Gemini)
- **Conflict resolution** — LLM-powered ADD/UPDATE/DELETE decisions
- **Context builder** — Token-budgeted, 3 strategies
- **Async API** — Full async support via AsyncMemBlock
- **Event hooks** — on_add, on_update, on_delete, on_query
- **Hierarchical scoping** — org → project → user → agent → session
- **Rerankers** — BM25, Cohere, CrossEncoder
- **Storage** — SQLite (local) or PostgreSQL (production)
- **CLI** — init, query, stats, prune, export, reindex

## Optional Extras

```bash
pip install "memblock[postgres]"            # PostgreSQL backend
pip install "memblock[embeddings]"          # Local vector embeddings (FastEmbed)
pip install "memblock[llm]"                 # LLM extraction (OpenAI, Anthropic, Gemini)
pip install "memblock[reranker-cohere]"     # Cohere reranker
pip install "memblock[reranker-cross-encoder]"  # HuggingFace reranker
pip install "memblock[all-cloud]"           # Everything without onnxruntime (Python 3.13+)
pip install "memblock[all]"                 # Everything including local embeddings
```

## Documentation

Full docs, API reference, and examples: **[memblock.xyz](https://memblock.xyz)**

## License

Proprietary. Copyright (c) 2025-2026 iexcalibur. All Rights Reserved.
