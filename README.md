# MemBlock

Structured, typed block tree + knowledge graph memory SDK for AI agents.

MemBlock gives developers full control over AI memory — what gets stored, how it's organized, how it decays, and how it's retrieved. No magic, no opinions, just a solid foundation.

## Why MemBlock?

Most "AI memory" tools auto-decide what to remember. That works until it doesn't — wrong memories, no encryption, no way to debug what went wrong.

MemBlock is different:

- **You decide** what to store (not the SDK)
- **5 block types**: FACT, PREFERENCE, EVENT, ENTITY, RELATION
- **Graph relationships**: 8 edge types (SUPPORTS, CONTRADICTS, CAUSED_BY, etc.)
- **Per-block encryption**: AES-256-GCM with 3 levels (NONE, STANDARD, SENSITIVE)
- **Tamper detection**: SHA-256 hash chain on every operation
- **Memory decay**: Exponential decay with access boost — old unused memories fade
- **Hybrid search**: FTS5 keyword search + vector embeddings + RRF merge
- **Token-budget context builder**: Feed relevant memories to your LLM within token limits
- **Multi-tenant PostgreSQL**: Production-ready with user_id scoping

## Install

```bash
# Core (FTS keyword search, encryption, graph, decay)
pip install memblock

# With local semantic search (no API key needed, ~80MB model download)
pip install memblock[embeddings]

# With OpenAI embeddings
pip install memblock[embeddings-openai]

# With LLM auto-extraction
pip install memblock[llm]

# PostgreSQL support
pip install memblock[postgres]

# Everything
pip install memblock[all]
```

## Quick Start

```python
from memblock import MemBlock, BlockType, SourceType, EdgeRelation

# Create a memory store (SQLite, local file)
mem = MemBlock(storage="sqlite:///./memory.db")

# Store memories
python_block = mem.store(
    content="User is a senior Python developer",
    type=BlockType.FACT,
    confidence=0.95,
    source=SourceType.EXPLICIT,
    tags=["skills", "python"],
)

django_block = mem.store(
    content="Prefers Django over Flask for web projects",
    type=BlockType.PREFERENCE,
    confidence=0.8,
    tags=["framework", "web"],
)

# Link them
mem.link(django_block.id, python_block.id, relation=EdgeRelation.ABOUT)

# Query
results = mem.query(type=BlockType.PREFERENCE, tags=["web"])
results = mem.query(text_search="Python developer")

# Build LLM context (stays within token budget)
context = mem.build_context(query="user tech preferences", token_budget=2000)
print(context)

# Check integrity
report = mem.verify()
print(report)  # TamperReport(valid=True, total_ops=4, ...)

mem.close()
```

## Semantic Search (Hybrid)

Enable embeddings for meaning-based search, not just keyword matching:

```python
# Local embeddings (FastEmbed, no API key)
mem = MemBlock(storage="sqlite:///./memory.db", embeddings=True)

# Or OpenAI embeddings
mem = MemBlock(
    storage="sqlite:///./memory.db",
    embeddings="openai",
    embeddings_api_key="sk-...",
)

# Store
mem.store("I enjoy building web apps with Django", type=BlockType.PREFERENCE)

# This finds the block even though "framework" isn't in the stored text
results = mem.query(text_search="favorite web framework")
```

Behind the scenes: FTS5 runs keyword search, vector cosine similarity runs semantic search, and Reciprocal Rank Fusion (RRF) merges both ranked lists into one.

## Encryption

Per-block AES-256-GCM encryption with three levels:

```python
mem = MemBlock(
    storage="sqlite:///./memory.db",
    encryption_key="your-secret-passphrase",
)

# Standard encryption
mem.store(
    content="User's API key is sk-abc123",
    type=BlockType.FACT,
    encryption_level=EncryptionLevel.STANDARD,
)

# Sensitive — higher security
mem.store(
    content="SSN: 123-45-6789",
    type=BlockType.FACT,
    encryption_level=EncryptionLevel.SENSITIVE,
)

# Retrieval auto-decrypts
block = mem.get(block_id)
print(block.content)  # plaintext
```

Each block gets a unique random nonce. Key derivation uses PBKDF2 with 480,000 iterations.

## Graph Relationships

```python
# 8 relationship types
mem.link(a.id, b.id, relation=EdgeRelation.SUPPORTS)
mem.link(a.id, b.id, relation=EdgeRelation.CONTRADICTS)
mem.link(a.id, b.id, relation=EdgeRelation.CAUSED_BY)
mem.link(a.id, b.id, relation=EdgeRelation.RELATED_TO)
mem.link(a.id, b.id, relation=EdgeRelation.PART_OF)
mem.link(a.id, b.id, relation=EdgeRelation.DERIVED_FROM)
mem.link(a.id, b.id, relation=EdgeRelation.SUPERSEDES)
mem.link(a.id, b.id, relation=EdgeRelation.ABOUT)

# Traverse the graph
neighbors = mem.neighbors(block.id)
connected = mem.traverse(block.id, max_depth=3)

# Query with graph context
results = mem.query(text_search="Python", related_to=entity_block.id)
```

## Memory Decay

Memories fade over time unless accessed. The formula:

```
strength = confidence × e^(-decay_rate × hours) × access_boost
```

```python
# Get strongest/weakest memories
strong = mem.strongest(limit=5)   # [(block, strength), ...]
weak = mem.weakest(limit=5)

# Auto-prune faded memories
pruned = mem.prune(min_strength=0.1)
```

## Context Builder

Build token-budgeted context strings for LLM prompts:

```python
# 3 strategies
context = mem.build_context(query="user preferences", token_budget=4000, strategy="relevance")
context = mem.build_context(query="user preferences", token_budget=4000, strategy="graph_walk")
context = mem.build_context(query="user preferences", token_budget=4000, strategy="type_grouped")
```

Output format:
```
## Memory Context
- [PREFERENCE] Prefers Django over Flask [HIGH] (explicit, strength=0.92) [framework, web]
- [FACT] User is a senior Python developer [HIGH] (explicit, strength=0.88) [skills, python]
```

## LLM Auto-Extraction

Extract structured memory blocks from conversations:

```python
# Requires: pip install memblock[llm]
result = mem.extract(
    conversation="User said they love Python and have been using Django for 3 years",
    provider="openai",
    api_key="sk-...",
)
print(result.block_ids)  # ['blk_...', 'blk_...']

# From message list
result = mem.extract_messages(
    messages=[
        {"role": "user", "content": "I've been coding in Rust lately"},
        {"role": "assistant", "content": "Rust is great for systems programming!"},
    ],
    provider="openai",
    api_key="sk-...",
)
```

## PostgreSQL (Multi-Tenant)

For production deployments with multiple users:

```python
# Requires: pip install memblock[postgres]
mem = MemBlock(
    storage="postgresql://user:pass@localhost:5432/mydb",
    user_id="user_123",
    embeddings=True,
)

# All blocks, edges, operations scoped to user_123
mem.store("User prefers dark mode", type=BlockType.PREFERENCE)

# Different user, same database
mem2 = MemBlock(
    storage="postgresql://user:pass@localhost:5432/mydb",
    user_id="user_456",
)
# mem2 sees only user_456's blocks
```

Tables are prefixed with `memblock_` and use composite primary keys `(id, user_id)`.

## Export & Stats

```python
# Export as markdown
md = mem.export_markdown()

# Get statistics
stats = mem.stats()
# {
#     "total_blocks": 42,
#     "deleted_blocks": 3,
#     "blocks_by_type": {"fact": 20, "preference": 12, "event": 10},
#     "total_edges": 28,
#     "total_operations": 87,
#     "embeddings_enabled": True,
#     "total_embeddings": 42,
# }
```

## Context Manager

```python
with MemBlock(storage="sqlite:///./memory.db") as mem:
    mem.store("auto-closes when done", type=BlockType.FACT)
# connection closed automatically
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                  MemBlock                     │  ← Public API
├──────────────────────────────────────────────┤
│  BlockStore  │ GraphIndex │ QueryEngine      │  ← Core components
│  OpLog       │ DecayEngine│ ContextBuilder   │
│  CryptoLayer │ Embeddings │ LLMExtractor     │
├──────────────────────────────────────────────┤
│         StorageAdapter (abstract)             │  ← Adapter pattern
├───────────────────┬──────────────────────────┤
│   SQLiteAdapter   │  PostgreSQLAdapter       │  ← Implementations
│   (local/dev)     │  (production/multi-user) │
└───────────────────┴──────────────────────────┘
```

## Block Types

| Type | Use For |
|------|---------|
| `FACT` | Objective information ("User is 28 years old") |
| `PREFERENCE` | Subjective choices ("Prefers dark mode") |
| `EVENT` | Things that happened ("Deployed v2.0 on March 1") |
| `ENTITY` | Named things ("Project Alpha", "John Smith") |
| `RELATION` | Connections between things ("John works at Acme") |

## Edge Relations

| Relation | Meaning |
|----------|---------|
| `ABOUT` | This block is about that entity |
| `SUPPORTS` | This evidence supports that claim |
| `CONTRADICTS` | This conflicts with that |
| `CAUSED_BY` | This was caused by that event |
| `RELATED_TO` | General relationship |
| `PART_OF` | This is a component of that |
| `DERIVED_FROM` | This was derived from that source |
| `SUPERSEDES` | This replaces that (newer info) |

## Tests

```bash
# Run all tests (207 tests)
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=memblock --cov-report=term-missing
```

## License

MIT
