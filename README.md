# MemBlock

Structured memory SDK for AI agents — full developer control over what gets stored, how it's organized, and how it's retrieved.

## Install

```bash
pip install memblock

# Optional extras
pip install memblock[embeddings]       # Local semantic search
pip install memblock[embeddings-openai] # OpenAI embeddings
pip install memblock[llm]              # LLM auto-extraction
pip install memblock[postgres]         # PostgreSQL support
pip install memblock[all]              # Everything
```

## Quick Start

```python
from memblock import MemBlock, BlockType, SourceType, EdgeRelation

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

# Link memories into a knowledge graph
mem.link(django_block.id, python_block.id, relation=EdgeRelation.ABOUT)

# Query
results = mem.query(type=BlockType.PREFERENCE, tags=["web"])
results = mem.query(text_search="Python developer")

# Build LLM context (stays within token budget)
context = mem.build_context(query="user tech preferences", token_budget=2000)

# Check integrity
report = mem.verify()

mem.close()
```

## Semantic Search

Enable embeddings for meaning-based search:

```python
# Local embeddings (no API key needed)
mem = MemBlock(storage="sqlite:///./memory.db", embeddings=True)

# Or OpenAI embeddings
mem = MemBlock(
    storage="sqlite:///./memory.db",
    embeddings="openai",
    embeddings_api_key="sk-...",
)

mem.store("I enjoy building web apps with Django", type=BlockType.PREFERENCE)

# Finds the block even though "framework" isn't in the stored text
results = mem.query(text_search="favorite web framework")
```

## Encryption

Per-block encryption with three levels:

```python
from memblock import EncryptionLevel

mem = MemBlock(
    storage="sqlite:///./memory.db",
    encryption_key="your-secret-passphrase",
)

mem.store(
    content="User's API key is sk-abc123",
    type=BlockType.FACT,
    encryption_level=EncryptionLevel.STANDARD,
)

mem.store(
    content="SSN: 123-45-6789",
    type=BlockType.FACT,
    encryption_level=EncryptionLevel.SENSITIVE,
)

# Retrieval auto-decrypts
block = mem.get(block_id)
print(block.content)  # plaintext
```

## Knowledge Graph

```python
# Link memories with typed relationships
mem.link(a.id, b.id, relation=EdgeRelation.SUPPORTS)
mem.link(a.id, b.id, relation=EdgeRelation.CONTRADICTS)
mem.link(a.id, b.id, relation=EdgeRelation.CAUSED_BY)

# Traverse the graph
neighbors = mem.neighbors(block.id)
connected = mem.traverse(block.id, max_depth=3)

# Query with graph context
results = mem.query(text_search="Python", related_to=entity_block.id)
```

## Memory Decay

Memories fade over time unless accessed. Prune weak ones automatically:

```python
strong = mem.strongest(limit=5)   # [(block, strength), ...]
weak = mem.weakest(limit=5)
pruned = mem.prune(min_strength=0.1)
```

## Context Builder

Build token-budgeted context strings for LLM prompts:

```python
context = mem.build_context(query="user preferences", token_budget=4000, strategy="relevance")
context = mem.build_context(query="user preferences", token_budget=4000, strategy="graph_walk")
context = mem.build_context(query="user preferences", token_budget=4000, strategy="type_grouped")
```

## Deduplication

Prevent duplicate memories with configurable policies:

```python
mem = MemBlock(storage="sqlite:///./memory.db", on_duplicate="error")           # Raise on duplicate
mem = MemBlock(storage="sqlite:///./memory.db", on_duplicate="skip")            # Silently skip
mem = MemBlock(storage="sqlite:///./memory.db", on_duplicate="return_existing") # Return original
mem = MemBlock(storage="sqlite:///./memory.db", on_duplicate="merge")           # Merge tags & confidence
```

## Auto-Extraction

Extract structured memories from conversations using an LLM:

```python
# One-shot extraction (requires: pip install memblock[llm])
result = mem.extract(
    conversation="User said they love Python and have been using Django for 3 years",
    provider="openai",
    api_key="sk-...",
)

# Opt-in streaming — buffer messages, extract periodically
mem = MemBlock(
    storage="sqlite:///./memory.db",
    auto_extract=True,
    extract_provider="openai",
    extract_api_key="sk-...",
    extract_every=10,
)
mem.add_message("user", "I've been coding in Rust lately")
mem.add_message("assistant", "Rust is great for systems programming!")
# ... extraction triggers every 10 messages, or flush manually:
result = mem.flush_extraction()
```

## PostgreSQL (Multi-Tenant)

```python
# Requires: pip install memblock[postgres]
mem = MemBlock(
    storage="postgresql://user:pass@localhost:5432/mydb",
    user_id="user_123",
    embeddings=True,
)

mem.store("User prefers dark mode", type=BlockType.PREFERENCE)

# Different user, same database — fully isolated
mem2 = MemBlock(
    storage="postgresql://user:pass@localhost:5432/mydb",
    user_id="user_456",
)
```

## CLI

```bash
memblock init --db sqlite:///./memory.db
memblock query "Python developer"
memblock query --type preference --tags web --json
memblock stats
memblock export --format json --output memories.json
memblock prune --min-strength 0.1 --dry-run
memblock reindex
memblock version
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

## License

MIT
