# MemBlock

**MemBlock** is a private, local-first memory SDK for AI applications.

It is built for teams and individuals who want fast, controllable memory without depending on a required cloud memory backend.

> Current distribution model: **invite-only / private access only** (not public PyPI).

---

## What is MemBlock?

MemBlock stores and retrieves context blocks for LLM-driven applications (assistants, copilots, automation agents) from your own database.  
It gives you:

- Reliable storage (SQLite / PostgreSQL)
- Deterministic retrieval APIs
- Deduplication (exact + semantic)
- Auto-extraction (optional)
- CLI utilities for ops
- Local license activation for controlled distribution

---

## Why MemBlock?

### 1) Local-first + owned data
Your data stays in your own DB. No dependency on a mandatory external memory service.

### 2) Developer-controlled
Everything runs through a Python SDK with a clear API surface and minimal infra assumptions.

### 3) Practical for production pipelines
Schema migration support, stable errors, tests, and CLI-first workflows make rollout easier than experimental scripts.

### 4) Cost-efficient
No forced cloud spend for baseline memory features.

### 5) Private and controlled distribution
Built for “approved users only” with install + runtime controls.

---

## Why choose MemBlock over common cloud memory defaults?

Compared with typical cloud-first memory layers, MemBlock is designed for control-first teams:

- **Data control:** your DB, your retention, your access policy
- **No vendor lock-in:** move/migrate using your own storage
- **Private deployment:** no open-source/public install path
- **Predictable behavior:** versioned storage + deterministic APIs
- **Configurable rules:** dedup, extraction, CLI actions, error handling

---

## How it works

1. You create blocks by storing content/messages.
2. Blocks are optionally de-duplicated before persisting.
3. Blocks are tagged/typed/confidence-scored and indexed in your DB.
4. You query by text/search filters for recall.
5. Optional background/interval extraction enriches memory from chat streams.
6. Runtime operations are validated via license state before use.

---

## Installation (invite-only)

MemBlock is distributed privately (example flows below):

### Private GitHub release asset
```bash
pip install "https://<TOKEN>@github.com/<org>/<repo>/releases/download/v0.2.0/memblock-0.2.0-py3-none-any.whl"
