# Changelog

## v0.13.1

### Fixed — package metadata

- **Corrected PyPI metadata.** The published `0.13.0` artifacts predated the metadata fix, so PyPI displayed `Author: None` and `License: Proprietary`. This release republishes with the correct `authors = [{name = "iexcalibur", …}]` and the **MIT** license (`license` field + `License :: OSI Approved :: MIT License` classifier). PyPI versions are immutable, so a new version is required to surface the corrected metadata. No code changes.
- **Synced `__version__`.** `memblock.__version__` was stale at `0.12.1` (never bumped for the `0.13.0` release); it now matches the package version.

## v0.13.0

### Added — externally-managed schema (run under a DML-only database role)

- **`manage_schema` flag on `MemBlock` and `AsyncMemBlock` (default `True`).** When `False`, the PostgreSQL adapters emit **zero DDL** — no `CREATE TABLE/INDEX/FUNCTION/TRIGGER`, no `ALTER`, no `CREATE SCHEMA`, and no migration runner. The memblock schema is expected to be provisioned out of band (see the new `sql/` directory), so the package can run under a least-privilege, DML-only database role. This satisfies security reviews that forbid an application dependency from holding DDL rights on the database. Threaded through to `PostgreSQLAdapter` / `AsyncPostgreSQLAdapter` via `manage_schema=`.

- **Every DDL surface is silenced, not just `initialize()`.** `manage_schema=False` also short-circuits the lazy pgvector index creation in `_ensure_pgvector_index()` (a runtime DDL path that otherwise fires on the first embedding write). Server-side pgvector search still works under the restricted role — `initialize()` runs only a **read-only** capability check (`SELECT … FROM pg_extension`) to detect the extension, never DDL. This is required because `initialize()` runs `CREATE OR REPLACE FUNCTION` unconditionally, which needs table ownership a DML-only role lacks; the flag is the supported way to avoid that call.

- **Standalone schema SQL (`sql/`).** `sql/01_memblock_schema.sql` is the full current-state schema (all migrations v2–v7 folded inline, `schema_version` seeded to 7), `sql/02_memblock_grants.sql` is a least-privilege DML-only `GRANT` set, and `sql/README.md` documents the DDL-vs-DML split, the deploy steps, and the runtime DDL surfaces to pre-provision. A DBA / migration tool (Flyway / Liquibase / Alembic) reviews and applies these instead of the package.

### Tests
- New: `test_manage_schema.py` — flag plumbing through both clients and adapters; the no-DDL invariant on `initialize()` and `_ensure_pgvector_index()` for both sync and async (spy connections, no DB required); a contrast test proving `manage_schema=True` still provisions; and a gated live-DB integration test proving a `manage_schema=False` client does full CRUD against a pre-provisioned schema while adding zero schema objects.

### Migration notes
- **Backward compatible.** `manage_schema` defaults to `True`, so existing deployments auto-provision exactly as before. The locked-down mode is strictly opt-in. PostgreSQL only; ignored for SQLite (a local file you own).

## v0.12.1

### Fixed

- **`AsyncMemBlock.introspect_user(include_deleted=True)` now actually surfaces soft-deleted blocks** (previously silently dropped them, regardless of the flag). Two bugs were present:
  1. The sync-fallback path went through `self._mem.query(...)`, which calls `query_blocks` and always strips `deleted=TRUE` rows. The `include_deleted` flag never reached the storage filter.
  2. The native-async path passed `{"include_deleted": False}` to `query_blocks`, but the storage filter key is `"deleted"` — the key mismatch meant the `deleted=FALSE` WHERE-clause was appended unconditionally, same silent-strip outcome.
  Both paths now route through `storage.get_all_blocks(include_deleted=...)`, the same primitive `MemBlock.introspect_user` (sync) already used correctly. Regression test added at `tests/test_hard_delete_and_introspect.py::TestAsyncIntrospectUser::test_async_introspect_include_deleted_returns_soft_deleted` — would have caught both bugs.

  Impact: "right to access" disclosure surfaces and any history-recovery flow (e.g. "show every prior value of this profile field") were silently empty in the async path. The sync path was correct throughout.

## v0.12.0

### Added — three gaps closed (G1, G2, G3) so MemBlock can power "AI remembers everything" use cases

- **Temporal-aware deduplication for EVENT blocks (G1).** Storing the same content at different `happened_at` timestamps now produces distinct blocks instead of colliding on the content-only hash. The right primitive for time-series state — daily XIRR/IIXR readings, portfolio-value snapshots, fund NAV history, anything where "same value at different times" is fundamentally two pieces of information. Per-type policy: only EVENT writes with an explicit `happened_at` use the temporal-aware hash; FACT / PREFERENCE / ENTITY / RELATION keep the existing content-only hash so re-stating the same fact still dedupes. New helper: `ContentHasher.hash_temporal(content, happened_at)`. Wired automatically inside `MemBlock.store()` and `AsyncMemBlock.store()` — callers don't change.

- **Hard delete (G2a).** `MemBlock.hard_delete(block_id)` and `AsyncMemBlock.hard_delete(block_id)` permanently purge a block + its edges + its embedding from storage. Distinct from the existing `delete()` (soft-delete: sets `deleted=True` and keeps the row for audit). Required for "right to be forgotten" under GDPR / India DPDP — soft-deleted blocks remain queryable with `include_deleted=True` and discoverable via op-log replay. Op-log entry is appended BEFORE the purge so the audit trail records the deletion even after the block row is gone. Batch variant: `hard_delete_many(block_ids)` returns the count actually purged (silently skips ids that didn't exist).

- **Introspection API (G2b).** `MemBlock.introspect_user()` and `AsyncMemBlock.introspect_user(*, user_id=None, include_deleted=False, limit=None)` return every block the SDK holds for the bound user — the canonical surface behind "what do you remember about me?" agent introspection AND privacy "right to access" disclosures. Excludes soft-deleted blocks by default so the disclosure matches what's actually used in advice. Sorted `created_at DESC` (newest first) for natural disclosure-UI layout. Pairs with `hard_delete_many()` for the full "show me everything → forget all of it" round trip.

- **RETRACT conflict action (G3).** New `ConflictActionType.RETRACT` value distinguishes user-initiated closures ("I sold my Vanguard position", "we cancelled the SIP", "moved out of Berlin") from factual corrections (DELETE — "actually I never lived in Berlin"). RETRACT soft-deletes the prior block AND writes a `CONTRADICTS` edge from the new retraction block to the closed-out block, preserving the "X was true until Y" history. The LLM conflict-resolver prompt was updated to teach the model when to emit RETRACT vs DELETE (closure verbs → RETRACT; "actually never" → DELETE). Wired into both sync and async `store()` paths under the existing `conflict_resolution=True` flag — no new opt-in needed.

### Changed
- `CONFLICT_SYSTEM_PROMPT` now documents the RETRACT action and the RETRACT-vs-DELETE distinction. Existing ADD / UPDATE / DELETE / NONE behavior unchanged.

### Tests
- New: `test_temporal_dedup.py` (pure-hash + storage-level tests covering same-content-different-time, same-content-same-time, FACT-still-dedupes, EVENT-without-happened_at-falls-back).
- New: `test_hard_delete_and_introspect.py` (hard_delete round-trip, soft-vs-hard distinction, hard_delete_many, introspect with/without include_deleted, introspect→forget round trip).
- New: `test_retract.py` (enum membership, prompt documents RETRACT, CONTRADICTS edge is written when `metadata._retracts` marker is set).

### Migration notes
- **Backward compatible.** No schema changes; the temporal hash is opt-in by block type and only fires when `happened_at` is provided. Existing FACT / PREFERENCE / ENTITY / RELATION writes behave identically.
- Existing soft-delete (`delete()`) is unchanged. `hard_delete()` is purely additive.
- Existing ConflictAction handlers can ignore the new `RETRACT` enum value — the SDK handles it transparently when conflict resolution is enabled. Custom handlers wishing to surface retracted state in the UI should look for `EdgeRelation.CONTRADICTS` edges out of newly-stored blocks.

## v0.11.0

### Added
- **Per-type decay defaults**: `MemBlock.store()` and `AsyncMemBlock.store()` now pick a sensible decay rate per `BlockType` when `decay_rate=None` (the new default). ENTITY blocks fade slowest (0.001), FACTs and RELATIONs slow (0.005), PREFERENCEs faster (0.020), EVENTs fastest (0.040). Reflects the durability of each kind of memory: named anchors don't change, opinions do. Caller-supplied floats still override. Tunable via `memblock.decay.DEFAULT_DECAY_BY_TYPE`.

- **Semantic-similarity auto-link edges**: when `enable_auto_link()` is on AND embeddings are configured, `store()` now adds a 3rd source of `RELATED_TO` edges — top-K nearest neighbours by embedding cosine similarity (in addition to sequential and tag-based edges). Skips matches below 0.5 similarity to avoid noise. Closes the gap where two semantically-related blocks shared no tags and weren't sequential. Capped at `_auto_link_max_neighbors` per write, best-effort, and silently skipped when embeddings aren't available.

- **Rule-based query expansion** (`memblock.query_expansion.expand_query`): a pure-regex helper that rewrites referential chat queries ("tell me more", "what about that", "compare them", "explain", "how does it work", etc.) using the most recent turn's content. Picks the longest capitalized noun phrase as the topic anchor, falls back to the longest lowercase noun phrase, returns the original query unchanged when no clear topic is found (never degrades a well-formed query). No LLM call, no extra cost, no extra latency. Designed for chat retrieval pipelines where vague follow-ups would otherwise return noise from semantic search.

### Tests
- 583 passing (was 558). New: `test_decay_per_type` (10 cases pinning the new defaults + override behaviour), `test_query_expansion` (15 cases covering referential detection, topic extraction, and "do not degrade well-formed queries").

### LoCoMo benchmark — for context

These v0.11.0 changes aren't visible on LoCoMo specifically (the benchmark uses fresh blocks so decay doesn't fire, doesn't enable auto-link, and asks well-formed questions so expansion doesn't apply). The proven LoCoMo numbers from v0.10.2 still hold:

```
                       Heuristic        BM25         Δ
─────────────────────────────────────────────────────────
  OVERALL recall          67.6%        90.2%      +22.6pp
  perfect rate            62.7%        85.8%      +23.0pp
  zero rate               27.5%         5.6%      -21.9pp
─────────────────────────────────────────────────────────
```

The v0.11.0 additions help in production scenarios the benchmark doesn't cover: long-running deployments where decay matters, chat follow-ups where expansion fires, and graphs where dense semantic edges power multi-hop walks.

## v0.10.2

### Fixed
- **Multi-hop relevance preservation**: `multi_hop_query()` was discarding the engine's relevance ranking. Hop-1 results were scored as `decay.calculate_strength(block) + 1.0` — a constant `+1.0` made all hop-1 candidates near-identical, and `decay.calculate_strength` (recency-biased) became the dominant tiebreaker. Result: multi-hop returned the most-recent blocks regardless of the query. Now uses **Reciprocal Rank weighting** (`1/(rank+1)`) so hop-1's relevance order is preserved through hop-2/hop-3 contributions. On the LoCoMo benchmark (199-question subset), recall jumped from **39.0% → 54.5%** (`+15.5pp`) with this fix alone. Sync `MultiHopRetriever` and async `AsyncMemBlock.multi_hop_query` both updated.

- **Type-scoped conflict resolution**: The LLM-driven conflict resolver was operating on type-mixed candidates — a `FACT` write semantically similar to an existing `ENTITY` block could trigger an `UPDATE` action that overwrote the entity's content with the fact's text. `ConflictResolver.resolve()` and `AsyncConflictResolver.aresolve()` now accept an optional `new_block_type=` parameter; when supplied, `UPDATE`/`DELETE` actions targeting blocks of a *different* type are dropped and the new write falls through to a regular `ADD`. Both `MemBlock.store()` and `AsyncMemBlock.store()` pass the new block's type, so the guard is on by default for all SDK consumers. Same-type conflict resolution (FACT→FACT, PREFERENCE→PREFERENCE, etc.) is unchanged.

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
