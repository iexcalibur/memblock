<p align="center">
  <h1 align="center">MemBlock</h1>
  <p align="center"><strong>Structured memory SDK for AI agents.</strong></p>
  <p align="center">Typed blocks · Knowledge graph · Hybrid search · Encryption · Decay engine — all local, all yours.</p>
  <p align="center"><a href="https://memblock.xyz">memblock.xyz</a></p>
</p>
<p align="center">
  <a href="https://pypi.org/project/memblock/"><img src="https://img.shields.io/pypi/v/memblock.svg" alt="PyPI version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://pypi.org/project/memblock/"><img src="https://img.shields.io/pypi/pyversions/memblock.svg" alt="Python versions"></a>
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

## Async (asyncio)

```python
from memblock import AsyncMemBlock, BlockType

# Native asyncpg path — non-blocking storage I/O
mem = AsyncMemBlock(storage="postgresql+asyncpg://user@host/db")

await mem.store("User prefers Python", type=BlockType.PREFERENCE)
results = await mem.query(text_search="programming", limit=10)

# Multi-tenant isolation: each tenant gets its own Postgres schema.
mem = AsyncMemBlock(
    storage="postgresql+asyncpg://user@host/db",
    schema="tenant_xyz",  # bootstraps + isolates on first use
)
```

`AsyncMemBlock` accepts plain `postgresql://` URLs too — those use the legacy thread-pool wrapper. Use `postgresql+asyncpg://` to opt into the native async backend.

## Optional Extras

```bash
pip install "memblock[postgres]"            # PostgreSQL backend (sync + async + pgvector)
pip install "memblock[embeddings]"          # Local vector embeddings (FastEmbed)
pip install "memblock[llm]"                 # LLM extraction (OpenAI, Anthropic, Gemini)
pip install "memblock[reranker-cohere]"     # Cohere reranker
pip install "memblock[reranker-cross-encoder]"  # HuggingFace reranker
pip install "memblock[all-cloud]"           # Everything without onnxruntime (Python 3.13+)
pip install "memblock[all]"                 # Everything including local embeddings
```

## Documentation

Full docs, API reference, and examples: **[memblock.xyz](https://memblock.xyz)**

## Contributing

Contributions are welcome! MemBlock is open source and community-driven.

- Found a bug or have a feature idea? [Open an issue](https://github.com/iexcalibur/memblock/issues).
- Want to contribute code? Fork the repo, create a branch, and open a pull request.
- Please make sure tests pass and follow the existing code style.

The `main` branch is protected, so all changes go through pull requests and review.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2025-2026 iexcalibur.
