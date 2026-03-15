# Changelog

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
