# Changelog

## v0.10.1

### Fixed
- **Native-mode reranker wiring**: `AsyncMemBlock(storage="postgresql+asyncpg://...", reranker=...)` now correctly forwards the reranker into `AsyncQueryEngine`. v0.10.0 accepted the `reranker=` kwarg silently but never wired it through in native-async mode, so reranking only worked for the legacy `to_thread` path. Any of `BM25Reranker`, `CohereReranker`, `CrossEncoderReranker`, `HeuristicReranker`, or `CallableReranker` now work uniformly across both modes.

### Added
- **`min_strength` and `include_decayed` query params** on both `MemBlock.query()` and `AsyncMemBlock.query()`. Lets callers filter out blocks below a decay-adjusted strength threshold at query time (not just at `prune()` time). Defaults preserve old behaviour (`min_strength=0.0`, `include_decayed=False`).
  ```python
  # Surface only strong recent memories
  blocks = await mem.query(text_search="retirement", min_strength=0.3)
  ```

## v0.10.0

### Added
- **Native asyncpg backend**: New `postgresql+asyncpg://` URL scheme for `AsyncMemBlock` runs storage I/O directly on `asyncpg` — no `asyncio.to_thread` wrapping. Full parity with the sync `MemBlock`: store, get, update, delete, query, build_context, link/unlink, neighbors, traverse, prune, strongest/weakest, verify, sessions, multi_hop_query, analytics. Same op-log hash chain, same pgvector dual-write strategy.
  ```python
  mem = AsyncMemBlock(storage="postgresql+asyncpg://user@host/db")
  ```
- **`AsyncPostgreSQLAdapter`**: `asyncpg.Pool`-backed adapter with `pgvector.asyncpg` codec registration. HNSW/IVFFlat index strategy auto-selected by embedding dimension.
- **`AsyncQueryEngine`**: native-async hybrid FTS + vector search with weighted RRF merging, graph-proximity boost, and concurrent BFS traversal via `asyncio.gather`.
- **`AsyncContextBuilder`**: native-async equivalent of `ContextBuilder` with all four strategies (`relevance`, `graph_walk`, `type_grouped`, `adaptive`).
- **`AsyncConflictResolver`**: native-async `aresolve()` for LLM-driven ADD/UPDATE/DELETE/NONE decisions.
- **`schema=` parameter**: `AsyncMemBlock(storage=..., schema='tenant_xyz')` bootstraps a custom Postgres schema on first use and keeps every write inside it. Drop the schema to clean up — `public` is untouched. Enables multi-tenant isolation without separate databases.
- **Hooks in async CRUD**: native `store`/`update`/`delete`/`query` emit `EventType.ON_ADD`/`ON_UPDATE`/`ON_DELETE`/`ON_QUERY` to registered callbacks (sync or async). Same lifecycle as sync `MemBlock`.
- **Async background extraction**: `AsyncMemBlock(auto_extract_on_store=True, background_extract=True)` schedules extraction via `asyncio.create_task` and tracks tasks for clean shutdown.
- **Test coverage**: 30 new tests in `tests/test_async_native.py` (URL detection, CRUD, query/FTS, edges, op-log, schema isolation, hooks). `pytest-asyncio` auto-mode configured in `pyproject.toml`.

### Changed
- **`[postgres]` extra now bundles both drivers**: `pip install "memblock[postgres]"` now installs `psycopg[binary]>=3.1` *and* `asyncpg>=0.29` *and* `pgvector>=0.3` — one extra covers `MemBlock`, `AsyncMemBlock` (legacy URL), and `AsyncMemBlock` (native URL). Previously `[postgres]` was sync-only.
- **`AsyncPostgreSQLAdapter.initialize()` creates the schema if missing**: `CREATE SCHEMA IF NOT EXISTS` runs before table DDL when `schema != "public"` so users can pass any custom schema name without pre-provisioning.

### Fixed
- **FTS trigger silently skipped on non-public schemas**: the `IF NOT EXISTS` check on `pg_trigger.tgname` is global per database, so once `public.memblock_blocks` had the trigger, every other schema short-circuited and ended up with `content_tsv` permanently NULL — breaking full-text search. Trigger existence check is now scoped to `tgrelid = '{schema}.memblock_blocks'::regclass`.
- **`AsyncMemBlockPool.get()` regression**: the pool builds `AsyncMemBlock` via `__new__()` to wrap an existing sync instance, which bypassed `__init__`. The newly-added `_native_async` attribute was never set, so every CRUD method on a pooled wrapper raised `AttributeError`. Pool now sets `_native_async = False` after `__new__`.
- **`tests/test_cli.py::test_version_output`**: hardcoded `"0.4.0"` assertion (stale across many releases) replaced with dynamic `__version__` check.

## v0.6.1

### Added
- **`[all-cloud]` install extra**: `pip install "memblock[all-cloud]"` installs all features except `fastembed` and `sentence-transformers` (which require `onnxruntime`). Works on Python 3.13+ where `onnxruntime` has no pre-built wheels. Includes cloud embeddings (OpenAI, Gemini), LLM extraction, PostgreSQL, pgvector, pooling, tokens, and Cohere reranker.

### Changed
- **`[all]` extra now includes `[all-cloud]`**: `[all]` is a superset of `[all-cloud]` plus local embeddings (`fastembed`) and cross-encoder reranker. Use `[all]` on Python ≤ 3.12 where `onnxruntime` is available.

## v0.4.3

### Changed
- **Updated PyPI README**: Added positioning copy and feature summary for better discoverability on PyPI.
- **Unified `llm` extra**: `pip install "memblock[llm]"` now includes OpenAI, Anthropic, and Gemini — no separate `llm-gemini` needed.

## v0.4.2

### Changed
- **Slim README for PyPI**: Reduced README to essentials with redirect to [memblock.xyz](https://memblock.xyz) for full documentation.
- **Full docs moved to `docs/FULL_DOCS.md`**: Detailed documentation preserved in repo but excluded from published package.
- **Updated project URLs**: Homepage and documentation now point to memblock.xyz.
- **Published to PyPI**: `pip install memblock` now works directly.

## v0.4.1

### Added
- **Gemini LLM Provider**: Google Gemini support for auto-extraction and conflict resolution (`extract_provider="gemini"`). Uses `google-genai` SDK with `gemini-2.0-flash` as default model.
- **`llm-gemini` install extra**: `pip install "memblock[llm-gemini]"` for Gemini-powered extraction.

## v0.2.0

### Added
- **Schema Migrations**: Automatic database schema versioning and migration system. Detects v0.1.0 databases and migrates forward. Future version guard prevents running against newer schemas.
- **Error Types**: Stable exception hierarchy — `MemBlockError`, `StorageError`, `MigrationError`, `ValidationError`, `BlockNotFoundError`, `DuplicateBlockError`, `ExtractionError`, `EncryptionError`.
- **Deduplication**: Content-hash based duplicate detection with four configurable policies: `error`, `skip`, `return_existing`, `merge`. Unicode-normalized SHA-256 hashing. Optional semantic dedup when embeddings are configured.
- **Auto-Extraction (Opt-in)**: Buffer conversation messages with `add_message()` and trigger LLM-based extraction at configurable intervals. Manual `flush_extraction()` for on-demand use. Buffer preserved on failure for retry.
- **CLI**: `memblock` command with subcommands: `init`, `query`, `stats`, `prune`, `export`, `reindex`, `version`.

### Changed
- `SchemaValidationError` is now a deprecated alias for `ValidationError` (backward compatible).
- `Block` dataclass includes `content_hash` field.

## v0.1.0

Initial release — typed block tree + knowledge graph memory for AI agents.

- SQLite and PostgreSQL storage adapters
- FTS5 full-text search
- Semantic search with embeddings (FastEmbed, OpenAI, Gemini)
- Hybrid search with Reciprocal Rank Fusion
- AES-256 encryption with field-level control
- Knowledge graph with typed edges
- Exponential decay engine
- Context builder with token budgets
- Operation log with SHA-256 hash chain for tamper detection
- LLM-based extraction (OpenAI, Anthropic)
