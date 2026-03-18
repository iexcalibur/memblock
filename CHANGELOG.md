# Changelog

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
