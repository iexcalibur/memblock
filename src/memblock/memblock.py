"""MemBlock — the main facade class composing all SDK components."""

from __future__ import annotations

import asyncio
import threading
import uuid
from enum import Enum
from typing import Any

from memblock.block import Block
from memblock.context import ContextBuilder
from memblock.crypto import CryptoLayerWithPassphrase
from memblock.decay import DecayEngine
from memblock.dedup import ContentHasher, DuplicateChecker, DuplicatePolicy
from memblock.errors import (
    AnalyticsError,
    DuplicateBlockError,
    EncryptionError,
    ExtractionError,
    LicenseError,
    TransactionError,
    ValidationError,
)
from memblock.background import BackgroundExtractor
from memblock.hooks import EventType, HookManager
from memblock.licensing import LicenseInfo, get_secret, get_stored_license, validate_license
from memblock.graph import GraphIndex
from memblock.ops import OpLog, TamperReport
from memblock.query import QueryEngine
from memblock.schema import BlockSchema, SchemaValidationError
from memblock.storage.base import StorageAdapter
from memblock.storage.sqlite import SQLiteAdapter
from memblock.store import BlockStore
from memblock.transactions import (
    DEFAULT_LEASE_SECONDS,
    RECEIPT_KEY,
    TXN_KEY,
    ReceiptStatus,
    Transaction,
    is_internal,
    is_staged,
    recover_transactions,
)
from memblock.types import (
    BlockType,
    EdgeRelation,
    EncryptionLevel,
    SourceType,
    now_utc,
)


class MemBlock:
    """
    Main entry point for the MemBlock SDK.

    Composes: BlockStore + GraphIndex + CryptoLayer + DecayEngine +
              QueryEngine + ContextBuilder + optional EmbeddingProvider
              into a single clean API.

    Usage:
        # SQLite (local) — keyword search only
        mem = MemBlock(storage="sqlite:///./memory.db")

        # SQLite with local embeddings (hybrid search)
        mem = MemBlock(storage="sqlite:///./memory.db", embeddings=True)

        # SQLite with OpenAI embeddings
        mem = MemBlock(
            storage="sqlite:///./memory.db",
            embeddings="openai",
            embeddings_api_key="sk-...",
        )

        # PostgreSQL (production, multi-user)
        mem = MemBlock(
            storage="postgresql://user:pass@localhost:5432/mydb",
            user_id="u_123",
            embeddings=True,
        )

        block = mem.store("User prefers Python", type=BlockType.PREFERENCE)
        mem.link(block.id, other.id, relation=EdgeRelation.SUPPORTS)
        results = mem.query(type=BlockType.PREFERENCE)
        context = mem.build_context(query="user preferences", token_budget=4000)
        mem.verify()
    """

    def __init__(
        self,
        storage: str = "sqlite:///:memory:",
        encryption_key: str | None = None,
        author: str = "agent",
        user_id: str = "default",
        embeddings: "bool | str | EmbeddingProvider" = False,
        embeddings_api_key: str | None = None,
        embeddings_model: str | None = None,
        on_duplicate: DuplicatePolicy | str | None = None,
        similarity_threshold: float = 0.95,
        auto_extract: bool = False,
        extract_provider: str | None = None,
        extract_api_key: str | None = None,
        extract_model: str | None = None,
        extract_every: int = 100,
        extract_min_confidence: float = 0.3,
        license_key: str | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        hooks: dict[str, list] | None = None,
        reranker: Any = None,
        auto_extract_on_store: bool = False,
        background_extract: bool = False,
        background_extract_workers: int = 2,
        conflict_resolution: bool = False,
        enable_analytics: bool = False,
        analytics_noise_words: set[str] | None = None,
        pool: Any = None,
        manage_schema: bool = True,
        classifier: Any = None,
        classifier_conflicts: bool = True,
        classifier_escalate: bool = True,
        on_embedding_mismatch: str = "error",
    ) -> None:
        """
        Initialize MemBlock.

        Args:
            storage: Storage URI. Supports:
                - "sqlite:///path/to/db.sqlite" (file-based)
                - "sqlite:///:memory:" (in-memory, default)
                - "postgresql://user:pass@host:port/db" (PostgreSQL)
                - "postgres://user:pass@host:port/db" (alias)
            encryption_key: Passphrase for AES-256 encryption. None = no encryption.
            author: Default author for operations.
            user_id: User ID for multi-tenant PostgreSQL deployments.
            embeddings: Enable embedding-based semantic search.
                - False: FTS only (default, no extra deps)
                - True: Local embeddings via FastEmbed (pip install memblock[embeddings])
                - "openai": OpenAI text-embedding-3-small (requires embeddings_api_key)
                - "gemini": Gemini gemini-embedding-001 (requires embeddings_api_key)
            embeddings_api_key: API key for OpenAI/Gemini embedding providers.
            embeddings_model: Override the default embedding model name.
            on_duplicate: Deduplication policy. None = no dedup (default).
                - "error": raise DuplicateBlockError
                - "skip": return None silently
                - "return_existing": return the existing block
                - "merge": merge tags/confidence, return updated block
            similarity_threshold: Cosine similarity threshold for semantic dedup (0.0-1.0).
            auto_extract: Enable opt-in auto-extraction from buffered messages.
            extract_provider: LLM provider for extraction ("openai", "anthropic", "gemini").
            extract_api_key: API key for extraction provider.
            extract_model: Model name override for extraction.
            extract_every: Trigger extraction every N messages (default 100).
            extract_min_confidence: Minimum confidence for extracted blocks (default 0.3).
            session_id: Default session ID for all operations. None = single-session mode
                (all blocks in one global scope). Set this or pass session_id per-call
                to opt into multi-session scoping.
            org_id: Default organization ID for hierarchical scoping.
            project_id: Default project ID for hierarchical scoping.
            agent_id: Default agent ID for hierarchical scoping.
            hooks: Pre-register event hooks. Dict mapping event names
                ("on_add", "on_update", "on_delete", "on_query") to lists of callbacks.
            reranker: Optional reranker for improving search quality.
                Use BM25Reranker() (zero deps), CohereReranker(api_key=...),
                or CrossEncoderReranker() from memblock.rerankers.
            auto_extract_on_store: When True and an LLM provider is configured,
                every store() call also runs extraction on the content to
                automatically derive additional facts, preferences, etc.
                Extracted blocks are linked via DERIVED_FROM to the original.
            background_extract: When True (with auto_extract_on_store=True),
                extraction runs in a background thread instead of blocking
                store(). store() returns instantly; extracted blocks appear
                asynchronously. Use wait_for_extractions() to block until done.
            background_extract_workers: Max concurrent extraction threads
                (default 2). Keep low to respect LLM rate limits.
            conflict_resolution: When True and an LLM provider + embeddings
                are configured, store() will first check for semantically
                similar existing blocks and use the LLM to decide whether
                to ADD, UPDATE, DELETE, or skip. Requires embeddings=True.
            enable_analytics: Enable org-level Q&A analytics (default False).
                When True, exposes log_question(), get_top_questions(), etc.
            analytics_noise_words: Additional noise words to filter from analytics
                (e.g. greetings, filler). Merged with built-in defaults.
            pool: Optional psycopg_pool.ConnectionPool for PostgreSQL connection
                pooling. When provided, the adapter obtains connections from the
                pool instead of creating its own. Ignored for SQLite storage.
                Install with: pip install memblock[pool]
            manage_schema: When True (default), the adapter provisions its own
                schema — CREATE TABLE/INDEX/FUNCTION/TRIGGER and migrations run
                on init. When False, the adapter performs NO DDL and assumes the
                memblock schema was provisioned out of band (see sql/ in the
                repo). Use False to run under a least-privilege, DML-only
                database role so an external package never issues DDL against
                your database. PostgreSQL only; ignored for SQLite (a local file
                you own). When False, server-side pgvector search still works —
                the adapter only runs a read-only capability check, not DDL.
            on_embedding_mismatch: What to do when this store's vectors were made
                by a different embedding model than the one configured here:
                "error" (default) raises EmbeddingMismatchError on first use,
                "warn" warns, "ignore" carries on (use it to call reembed()).
        """
        # ── License validation (disabled — re-enable for paid tier) ──
        # secret = get_secret()
        # key = license_key or get_stored_license()
        # if key is not None and secret is not None:
        #     self._license: LicenseInfo = validate_license(key, secret)
        # elif secret is not None:
        #     raise LicenseError(
        #         "No valid license found. Run: memblock activate <key>"
        #     )
        # else:
        #     self._license = None
        self._license = None  # type: ignore[assignment]

        self._user_id = user_id
        self._session_id = session_id
        self._org_id = org_id
        self._project_id = project_id
        self._agent_id = agent_id

        # Event hooks
        self._hooks = HookManager()
        if hooks:
            for event_name, callbacks in hooks.items():
                for cb in callbacks:
                    self._hooks.register(EventType(event_name), cb)

        # Parse storage URI
        self._pool = pool
        self._storage = self._create_storage(
            storage, user_id, pool=pool, manage_schema=manage_schema
        )
        self._storage.initialize()

        # Initialize embedding provider (optional)
        self._embedding_provider = self._create_embedding_provider(
            embeddings, embeddings_api_key, embeddings_model
        )

        # Compose components
        self._crypto = CryptoLayerWithPassphrase(key=encryption_key)
        self._store = BlockStore(self._storage, author=author)
        self._graph = GraphIndex(self._storage, self._store.op_log)
        self._decay = DecayEngine(self._storage)
        self._reranker = reranker
        self._query = QueryEngine(
            self._storage, self._graph, self._decay,
            embedding_provider=self._embedding_provider,
            reranker=self._reranker,
        )
        self._context = ContextBuilder(self._storage, self._query, self._graph, self._decay)

        # Deduplication
        if isinstance(on_duplicate, str):
            on_duplicate = DuplicatePolicy(on_duplicate)
        self._on_duplicate = on_duplicate
        self._dedup = DuplicateChecker(
            self._storage,
            embedding_provider=self._embedding_provider,
            similarity_threshold=similarity_threshold,
        ) if on_duplicate is not None else None

        # Auto-extraction
        self._auto_extract = auto_extract
        self._extract_provider_name = extract_provider
        self._extract_api_key = extract_api_key
        self._extract_model = extract_model
        self._extract_every = extract_every
        self._extract_min_confidence = extract_min_confidence
        self._message_buffer: list[dict[str, str]] = []
        self._message_count: int = 0
        self._extractor = None  # Lazy-initialized
        self._auto_extract_on_store = auto_extract_on_store
        self._background_extract = background_extract and auto_extract_on_store
        self._bg_extractor: BackgroundExtractor | None = None
        if self._background_extract:
            self._bg_extractor = BackgroundExtractor(
                max_workers=background_extract_workers,
            )
        self._conflict_resolution = conflict_resolution
        # Pluggable memory classifier (see memblock/classifier.py). When set, store() fills
        # type/source/confidence/encryption_level/decay_rate the caller left unset from the
        # classifier's decision, and conflict resolution consults it before the LLM resolver.
        self._classifier = classifier
        self._classifier_conflicts = bool(classifier_conflicts)
        self._classifier_escalate = bool(classifier_escalate)
        self._conflict_resolver = None  # Lazy-initialized
        self._extracting = False  # Prevents infinite recursion (sync mode, main thread)
        self._thread_local = threading.local()  # Per-thread extracting state

        # Grouped writes (memblock/transactions.py): the open transaction is
        # per thread, so another thread's writes are never staged by accident.
        self._txn_local = threading.local()
        self._txn_recovered = False
        self._instance_id = uuid.uuid4().hex

        # Embedding identity (memblock/snapshot.py): checked on first use.
        if on_embedding_mismatch not in ("error", "warn", "ignore"):
            raise ValueError("on_embedding_mismatch must be 'error', 'warn' or 'ignore'")
        self._on_embedding_mismatch = on_embedding_mismatch
        self._embedding_identity_checked = False

        # Auto-linking: automatically build knowledge graph edges during store()
        self._auto_link = False
        self._auto_link_max_neighbors = 5
        self._last_stored_id: str | None = None  # for sequential linking
        self._tag_index: dict[str, list[str]] = {}  # tag -> list of recent block IDs

        # Analytics (opt-in)
        self._enable_analytics = enable_analytics
        self._analytics_noise_words = analytics_noise_words
        self._analytics = None  # Lazy-initialized

    @staticmethod
    def _create_storage(
        uri: str,
        user_id: str = "default",
        pool: Any = None,
        manage_schema: bool = True,
    ) -> StorageAdapter:
        """Create a storage adapter from a URI string."""
        if uri.startswith("postgresql://") or uri.startswith("postgres://"):
            try:
                from memblock.storage.postgresql import PostgreSQLAdapter
            except ImportError:
                raise ImportError(
                    "PostgreSQL adapter requires psycopg. "
                    "Install with: pip install memblock[postgres]"
                )
            return PostgreSQLAdapter(
                dsn=uri, user_id=user_id, pool=pool, manage_schema=manage_schema
            )
        elif uri.startswith("sqlite:///"):
            db_path = uri[len("sqlite:///"):]
            return SQLiteAdapter(db_path)
        elif uri.startswith("sqlite://"):
            db_path = uri[len("sqlite://"):]
            return SQLiteAdapter(db_path)
        else:
            # Default to SQLite with the URI as path
            return SQLiteAdapter(uri)

    @staticmethod
    def _create_embedding_provider(
        embeddings: bool | str,
        api_key: str | None,
        model: str | None,
    ) -> Any:
        """Create an embedding provider based on the embeddings parameter."""
        if embeddings is False:
            return None

        from memblock.embeddings import EmbeddingProvider
        if isinstance(embeddings, EmbeddingProvider):
            # A provider the application brings (e.g. a local model behind its own server)
            return embeddings

        if embeddings is True:
            # Local embeddings via FastEmbed
            try:
                from memblock.embeddings import FastEmbedProvider
                return FastEmbedProvider(model=model or "sentence-transformers/all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "Local embeddings require fastembed. "
                    "Install with: pip install memblock[embeddings]"
                )

        if isinstance(embeddings, str):
            if embeddings.lower() == "openai":
                if not api_key:
                    raise ValueError("embeddings_api_key is required for OpenAI embeddings")
                from memblock.embeddings import OpenAIEmbeddingProvider
                return OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model=model or "text-embedding-3-small",
                )
            elif embeddings.lower() == "gemini":
                if not api_key:
                    raise ValueError("embeddings_api_key is required for Gemini embeddings")
                from memblock.embeddings import GeminiEmbeddingProvider
                return GeminiEmbeddingProvider(
                    api_key=api_key,
                    model=model or "gemini-embedding-001",
                )
            else:
                raise ValueError(
                    f"Unknown embeddings provider: {embeddings}. "
                    "Use True (local), 'openai', or 'gemini'."
                )

        return None

    # ─── Event Hooks ──────────────────────────────────────────────────────

    def on(self, event: str, callback: Any) -> None:
        """
        Register a callback for a memory lifecycle event.

        Args:
            event: "on_add", "on_update", "on_delete", or "on_query"
            callback: Function that receives a dict with event data.
                Sync: def callback(data: dict) -> None
                Async: async def callback(data: dict) -> None
        """
        import inspect
        if inspect.iscoroutinefunction(callback):
            self._hooks.register_async(EventType(event), callback)
        else:
            self._hooks.register(EventType(event), callback)

    def enable_auto_link(
        self,
        enabled: bool = True,
        max_neighbors: int = 5,
    ) -> None:
        """
        Enable automatic knowledge graph linking during store().

        When enabled, each store() call will automatically create graph edges:
        1. Sequential link: RELATED_TO edge to the previously stored block
        2. Tag-based links: RELATED_TO edges to recent blocks sharing tags
           (e.g. same session, same speaker)

        This builds a rich knowledge graph without requiring an LLM,
        enabling multi-hop retrieval and graph-walk context strategies.

        Args:
            enabled: Turn auto-linking on/off
            max_neighbors: Max tag-based edges per store() call (default 5)
        """
        self._auto_link = enabled
        self._auto_link_max_neighbors = max_neighbors
        if enabled:
            self._last_stored_id = None
            self._tag_index = {}

    def _auto_link_block(self, block: Block, tags: list[str] | None) -> None:
        """Create automatic graph edges for a newly stored block."""
        # Retraction edges fire regardless of `_auto_link`. The
        # conflict resolver explicitly opted in by marking
        # `metadata._retracts: [block_ids]` on this block during
        # store(); writing the CONTRADICTS edge is the materialisation
        # of that intent. Skipping it because auto-link is off would
        # silently break the retraction semantic.
        retracts = (
            getattr(block.metadata, "custom_metadata", None) or {}
        ).get("_retracts") if hasattr(block, "metadata") else None
        if retracts:
            for retracted_id in retracts:
                try:
                    self._graph.link(
                        block.id, retracted_id,
                        relation=EdgeRelation.CONTRADICTS, weight=1.0,
                    )
                except Exception:
                    pass

        if not self._auto_link:
            return

        linked_ids: set[str] = set()

        # 1. Sequential link: connect to the previous block
        if self._last_stored_id is not None and self._last_stored_id != block.id:
            try:
                self._graph.link(
                    block.id, self._last_stored_id,
                    relation=EdgeRelation.RELATED_TO, weight=0.8,
                )
                linked_ids.add(self._last_stored_id)
            except Exception:
                pass

        # 2. Tag-based links: connect to recent blocks with shared tags
        if tags:
            candidates: dict[str, int] = {}  # block_id -> shared_tag_count
            for tag in tags:
                for prev_id in self._tag_index.get(tag, []):
                    if prev_id != block.id and prev_id not in linked_ids:
                        candidates[prev_id] = candidates.get(prev_id, 0) + 1

            # Sort by number of shared tags (more shared = stronger connection)
            sorted_candidates = sorted(
                candidates.items(), key=lambda x: x[1], reverse=True
            )

            for prev_id, shared_count in sorted_candidates[:self._auto_link_max_neighbors]:
                weight = min(1.0, 0.3 + shared_count * 0.2)  # 0.5 for 1 tag, 0.7 for 2, etc.
                try:
                    self._graph.link(
                        block.id, prev_id,
                        relation=EdgeRelation.RELATED_TO, weight=weight,
                    )
                    linked_ids.add(prev_id)
                except Exception:
                    pass

            # Update tag index (keep last 50 per tag to avoid memory bloat)
            for tag in tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = []
                self._tag_index[tag].append(block.id)
                if len(self._tag_index[tag]) > 50:
                    self._tag_index[tag] = self._tag_index[tag][-50:]

        self._last_stored_id = block.id

    # ─── Store Operations ─────────────────────────────────────────────────

    def store(
        self,
        content: str,
        type: BlockType | None = None,
        confidence: float | None = None,
        source: SourceType | None = None,
        tags: list[str] | None = None,
        parent_id: str | None = None,
        encryption_level: EncryptionLevel | None = None,
        decay_rate: float | None = None,
        ttl: int | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        happened_at: Any | None = None,
        happened_at_end: Any | None = None,
        temporal_precision: str = "exact",
        receipt: str | None = None,
    ) -> Block:
        """
        Store a new memory block.

        Args:
            content: The memory content text
            type: Block type (FACT, PREFERENCE, EVENT, ENTITY, RELATION).
                Defaults to FACT, or to the classifier's decision when
                `MemBlock(classifier=...)` is set and this is left unset.
            confidence: Confidence score (0.0-1.0); default 1.0 or the classifier's calibrated value
            source: How the memory was acquired; default EXPLICIT or the classifier's decision
            tags: Categorization tags
            parent_id: Parent block ID for tree structure
            encryption_level: NONE, STANDARD, or SENSITIVE
            decay_rate: How fast this memory fades (0 = never).
                When `None` (default), the SDK picks a type-specific
                rate from `decay.DEFAULT_DECAY_BY_TYPE`:
                ENTITY=0.001, FACT=0.005, RELATION=0.005,
                PREFERENCE=0.020, EVENT=0.040.
            ttl: Time-to-live in seconds (None = permanent)
            session_id: Session scope (overrides default)
            org_id: Organization scope (overrides default)
            project_id: Project scope (overrides default)
            agent_id: Agent scope (overrides default)
            metadata: Arbitrary key-value metadata for filtering
            receipt: A caller-chosen ID for this write. Storing again with a
                receipt that was already applied returns the block written the
                first time and writes nothing, so a retry cannot store twice.

        Inside `with mem.transaction():` the block is written staged: it becomes
        visible, together with the rest of the transaction, on publish.

        Returns:
            The created Block, or None if duplicate with SKIP policy.
        """
        txn = self._current_txn()
        if receipt is not None:
            replay = self._receipt_replay(receipt, txn)
            if replay is not None:
                return replay

        # ── Pluggable classification (memblock/classifier.py) ──────────────────
        # Fills whatever the caller left unset. Explicit arguments always win.
        classifier_record: dict[str, Any] | None = None
        if self._classifier is not None and not self._extracting and content:
            (type, confidence, source, encryption_level, decay_rate, temporal_precision,
             tags, classifier_record, skipped) = self._classify_for_store(
                content, type, confidence, source, encryption_level, decay_rate, temporal_precision, tags,
            )
            if skipped:
                return None  # type: ignore[return-value]
            if classifier_record is not None:
                from memblock.classifier import CLASSIFIER_METADATA_KEY
                metadata = dict(metadata or {})
                metadata[CLASSIFIER_METADATA_KEY] = classifier_record

        if type is None:
            type = BlockType.FACT
        if confidence is None:
            confidence = 1.0
        if source is None:
            source = SourceType.EXPLICIT
        if encryption_level is None:
            encryption_level = EncryptionLevel.NONE

        # Resolve per-type decay default when caller didn't override.
        if decay_rate is None:
            from memblock.decay import default_decay_rate_for
            decay_rate = default_decay_rate_for(type)

        # Conflict resolution: classifier first (fast, local), LLM resolver as the fallback
        # for escalated or unsupported decisions.
        llm_conflicts_possible = (
            self._embedding_provider is not None and self._extract_provider_name is not None
        )
        classifier_conflicts_possible = (
            self._classifier is not None and self._classifier_conflicts
            and hasattr(self._classifier, "resolve_conflict")
        )
        if (
            self._conflict_resolution
            and not self._extracting
            and (llm_conflicts_possible or classifier_conflicts_possible)
        ):
            try:
                similar = self.query(
                    text_search=content,
                    semantic=self._embedding_provider is not None,
                    limit=5,
                    session_id=session_id or self._session_id,
                )
                if similar:
                    from memblock.conflict import ConflictAction, ConflictResolver, ConflictActionType
                    actions: list[Any] = []
                    decided_by = None
                    if classifier_conflicts_possible:
                        decisions = self._classifier.resolve_conflict(content, similar)
                        if decisions and not any(d.get("escalate") for d in decisions):
                            actions = [
                                ConflictAction(action=ConflictActionType(d["action"]), block_id=d.get("block_id"),
                                               new_content=content, reason="classifier")
                                for d in decisions if d.get("action") != "ADD"
                            ]
                            decided_by = "classifier"
                        if classifier_record is not None and decisions:
                            classifier_record["conflict"] = [
                                {k: v for k, v in d.items() if k != "record"} for d in decisions
                            ]
                    if decided_by is None and llm_conflicts_possible:
                        if self._conflict_resolver is None:
                            extractor_provider = self._init_extractor().provider
                            self._conflict_resolver = ConflictResolver(provider=extractor_provider)
                        result = self._conflict_resolver.resolve(
                            content, similar, new_block_type=type,
                        )
                        actions = result.actions
                        decided_by = "llm"
                    if classifier_record is not None and decided_by:
                        classifier_record["conflict_decided_by"] = decided_by
                        metadata = dict(metadata or {})
                        metadata[CLASSIFIER_METADATA_KEY] = classifier_record
                    for action in actions:
                        if action.action == ConflictActionType.UPDATE and action.block_id:
                            self.update(action.block_id, content=action.new_content or content)
                            return self.get(action.block_id)  # type: ignore[return-value]
                        elif action.action == ConflictActionType.DELETE and action.block_id:
                            self.delete(action.block_id)
                        elif action.action == ConflictActionType.RETRACT and action.block_id:
                            # Retraction — soft-delete the prior block
                            # AND record a CONTRADICTS edge from the
                            # incoming retraction statement (which we
                            # ADD below) back to the closed-out block.
                            # We don't have the new block id yet (we're
                            # about to ADD it), so we delete first and
                            # link inside `_auto_link_block` via a
                            # marker on the retraction's metadata.
                            self.delete(action.block_id)
                            # The auto-link layer reads this marker to
                            # write the CONTRADICTS edge from the
                            # newly-created retraction block to the
                            # block we just closed. See `_auto_link_block`.
                            if metadata is None:
                                metadata = {}
                            metadata = dict(metadata)
                            metadata.setdefault("_retracts", []).append(action.block_id)
                            # ADD falls through with metadata enriched
                        elif action.action == ConflictActionType.NONE:
                            return None  # type: ignore[return-value]
                        # ADD falls through to normal store flow
            except Exception:
                pass  # conflict resolution failure falls through to normal store

        # Check for duplicates before creating.
        #
        # Temporal-aware hashing for EVENT blocks: the same content
        # at different `happened_at` timestamps must coexist (e.g.
        # daily portfolio-value snapshots, XIRR/IIXR readings over
        # time). For non-EVENT types we keep the content-only hash
        # so re-stating the same fact still dedupes. See the
        # docstring on `ContentHasher.hash_temporal` for the
        # rationale on per-type dedup keys.
        if self._dedup is not None and self._on_duplicate is not None:
            if type == BlockType.EVENT and happened_at is not None:
                content_hash = ContentHasher.hash_temporal(content, happened_at)
            else:
                content_hash = ContentHasher.hash(content)
            existing = self._dedup.check(content, content_hash)
            if existing is not None:
                if self._on_duplicate == DuplicatePolicy.ERROR:
                    raise DuplicateBlockError(
                        f"Duplicate content detected (block {existing.id})"
                    )
                elif self._on_duplicate == DuplicatePolicy.SKIP:
                    return None  # type: ignore[return-value]
                elif self._on_duplicate == DuplicatePolicy.RETURN_EXISTING:
                    return existing
                elif self._on_duplicate == DuplicatePolicy.MERGE:
                    # Merge tags and take higher confidence
                    merged_tags = list(set(existing.tags + (tags or [])))
                    merged_confidence = max(existing.metadata.confidence, confidence)
                    self.update(
                        existing.id,
                        tags=merged_tags,
                        confidence=merged_confidence,
                    )
                    existing.tags = merged_tags
                    existing.metadata.confidence = merged_confidence
                    return existing

        block = self._create_block(
            content,
            type=type,
            confidence=confidence,
            source=source,
            tags=tags,
            parent_id=parent_id,
            encryption_level=encryption_level,
            decay_rate=decay_rate,
            ttl=ttl,
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata=metadata,
            happened_at=happened_at,
            happened_at_end=happened_at_end,
            temporal_precision=temporal_precision,
            receipt=receipt,
            txn=txn,
        )

        def _announce() -> None:
            # Fire on_add hooks
            self._hooks.emit(EventType.ON_ADD, {
                "block_id": block.id,
                "block": block,
                "content": content,
                "type": type.value,
            })
            # Auto-link: build knowledge graph edges automatically
            self._auto_link_block(block, tags)

        # A staged block is not visible yet: hooks and the auto-linker hear
        # about it when its transaction publishes.
        if txn is not None:
            txn._on_publish(_announce)
        else:
            _announce()

        # Auto-extract additional memories from the content
        # Use thread-local flag to prevent recursive extraction
        # (extraction stores blocks → store() would re-trigger extraction)
        _is_extracting = getattr(self._thread_local, 'extracting', False)
        if (
            self._auto_extract_on_store
            and not _is_extracting
            and not self._extracting
            and self._extract_provider_name is not None
        ):
            if self._background_extract and self._bg_extractor is not None and txn is None:
                # Background mode: submit to thread pool, return immediately
                def _extract_fn(text: str) -> Any:
                    self._thread_local.extracting = True
                    try:
                        extractor = self._init_extractor()
                        return extractor.extract(text, memblock=self)
                    finally:
                        self._thread_local.extracting = False

                def _link_fn(extracted_id: str, source_id: str) -> None:
                    self.link(extracted_id, source_id, relation=EdgeRelation.DERIVED_FROM)

                self._bg_extractor.submit(
                    block_id=block.id,
                    content=content,
                    extract_fn=_extract_fn,
                    link_fn=_link_fn,
                )
            else:
                # Synchronous mode: block until extraction completes
                self._extracting = True
                try:
                    extractor = self._init_extractor()
                    result = extractor.extract(content, memblock=self)
                    # Link extracted blocks to the original via DERIVED_FROM
                    for extracted_id in (result.block_ids if result else []):
                        try:
                            self.link(extracted_id, block.id, relation=EdgeRelation.DERIVED_FROM)
                        except Exception:
                            pass
                except Exception:
                    pass  # extraction failure should not break store
                finally:
                    self._extracting = False

        return block

    def get(self, block_id: str, decrypt: bool = True) -> Block | None:
        """
        Retrieve a block by ID.

        Automatically decrypts content if encrypted and key is available.
        """
        block = self._store.get(block_id)
        if block is None:
            return None

        if decrypt and block.encrypted and self._crypto.enabled:
            try:
                block.content = self._crypto.open(block.content, block.encryption_level)
            except Exception as e:
                raise EncryptionError(f"Failed to decrypt block {block_id}: {e}") from e

        return block

    def update(self, block_id: str, **updates: Any) -> Block | None:
        """Update a block's fields.

        Inside a transaction a change to an existing block is held back and
        applied on publish; the block returned is the one as it stands now.
        """
        txn = self._current_txn()
        if txn is not None and block_id not in txn._staged:
            return self._defer_update(txn, block_id, updates)
        # If updating content and block is encrypted, encrypt the new content
        original_content = updates.get("content")
        if "content" in updates:
            block = self._store.get_without_touch(block_id)
            if block and block.encrypted and self._crypto.enabled:
                updates["content"] = self._crypto.seal(
                    updates["content"], block.encryption_level
                )

        result = self._store.update(block_id, **updates)

        # Re-embed if content changed
        if original_content is not None and self._embedding_provider is not None:
            self._embed_block(block_id, original_content)

        # Fire on_update hooks
        if result is not None:
            self._hooks.emit(EventType.ON_UPDATE, {
                "block_id": block_id,
                "block": result,
                "updates": updates,
            })

        return result

    def delete(self, block_id: str, cascade: bool = False) -> bool:
        """Soft-delete a block. Inside a transaction it is applied on publish."""
        txn = self._current_txn()
        if txn is not None:
            return self._defer_delete(txn, block_id, cascade)
        result = self._store.delete(block_id, cascade=cascade)
        # Remove embedding on delete
        if result:
            self._storage.delete_embedding(block_id)
            # Fire on_delete hooks
            self._hooks.emit(EventType.ON_DELETE, {
                "block_id": block_id,
                "cascade": cascade,
            })
        return result

    def hard_delete(self, block_id: str) -> bool:
        """Permanently purge a block — see `AsyncMemBlock.hard_delete`
        for the contract. Irreversible. Required for GDPR / India
        DPDP "right to be forgotten" — soft-delete keeps the row
        for audit; only hard-delete removes the data."""
        if self._current_txn() is not None:
            raise TransactionError(
                "hard_delete() is permanent and cannot be part of a transaction"
            )
        block = self._storage.get_block(block_id)
        if block is None:
            return False
        result = self._store.hard_delete(block_id)
        if result:
            try:
                self._storage.delete_embedding(block_id)
            except Exception:
                pass
            self._hooks.emit(EventType.ON_DELETE, {
                "block_id": block_id,
                "block": block,
                "cascade": False,
                "hard": True,
            })
        return result

    def hard_delete_many(self, block_ids: list[str]) -> int:
        """Batch hard-delete. Returns the count actually purged."""
        count = 0
        for bid in block_ids:
            try:
                if self.hard_delete(bid):
                    count += 1
            except Exception:
                continue
        return count

    def introspect_user(
        self,
        *,
        include_deleted: bool = False,
        limit: int | None = None,
    ) -> list["Block"]:
        """Return every block this instance holds for its bound user.
        Powers the "what do you remember about me?" surface — see the
        AsyncMemBlock variant for the full contract / use cases.

        Goes directly to the storage layer (not through `query()`)
        because `query()` applies decay + relevance ranking + always
        excludes soft-deleted blocks. Introspection needs the raw
        set, optionally including soft-deleted rows for audit /
        disclosure flows.
        """
        blocks = [
            b for b in self._storage.get_all_blocks(include_deleted=include_deleted)
            if not is_internal(b) and not is_staged(b)
        ]
        # Sort newest-first for the disclosure UI.
        try:
            blocks.sort(
                key=lambda b: getattr(b.metadata, "created_at", None) or "",
                reverse=True,
            )
        except Exception:
            pass
        if limit is not None:
            blocks = blocks[:limit]
        return blocks

    # ─── Graph Operations ─────────────────────────────────────────────────

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | str = EdgeRelation.RELATED_TO,
        weight: float = 1.0,
    ) -> None:
        """Create a relationship between two blocks."""
        if isinstance(relation, str):
            relation = EdgeRelation(relation)
        txn = self._current_txn()
        if txn is not None:
            self._defer_link(txn, source_id, target_id, relation, weight)
            return
        self._graph.link(source_id, target_id, relation, weight)

    def unlink(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | str | None = None,
    ) -> int:
        """Remove relationship(s) between two blocks."""
        if isinstance(relation, str):
            relation = EdgeRelation(relation)
        txn = self._current_txn()
        if txn is not None:
            txn._defer({
                "op": "unlink",
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation.value if relation is not None else None,
            })
            return 0  # the count is known once the transaction publishes
        return self._graph.unlink(source_id, target_id, relation)

    def neighbors(self, block_id: str, relation: EdgeRelation | None = None) -> list[Block]:
        """Get blocks directly connected to a block."""
        return self._graph.neighbors(block_id, relation=relation)

    def traverse(self, block_id: str, max_depth: int = 3) -> list[Block]:
        """Walk the graph from a block, returning all connected blocks."""
        return self._graph.traverse(block_id, max_depth=max_depth)

    # ─── Query ────────────────────────────────────────────────────────────

    def query(
        self,
        type: BlockType | None = None,
        tags: list[str] | None = None,
        text_search: str | None = None,
        related_to: str | None = None,
        min_confidence: float = 0.0,
        sort_by: str = "relevance",
        limit: int = 10,
        include_decayed: bool = False,
        min_strength: float = 0.0,
        semantic: bool = True,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        time_window: str = "rank",
    ) -> list[Block]:
        """
        Query memory blocks with structured filters.

        Args:
            type: Filter by block type
            tags: Filter by tags (match any)
            text_search: Full-text search (FTS + optional vector hybrid)
            related_to: Block ID — find graph-connected blocks
            min_confidence: Minimum confidence threshold
            sort_by: 'relevance', 'recency', 'access_count', 'strength'.
                With text_search the candidates are the top max(limit*5, 50)
                keyword-ranked matches (newest / most-accessed matches for
                'recency' / 'access_count'); 'strength' considers every match.
            limit: Maximum results
            include_decayed: Include blocks below `min_strength` (default False)
            min_strength: Minimum decayed-strength score (0.0-1.0) — filters
                stale/low-confidence blocks at query time. Default 0.0
                (off). Useful values: 0.1 to skip very weak blocks,
                0.3 to focus on strong recent ones.
            semantic: Enable hybrid search when embeddings are available (default True)
            session_id: Filter by session (overrides default)
            org_id: Filter by organization (overrides default)
            project_id: Filter by project (overrides default)
            agent_id: Filter by agent (overrides default)
            metadata_filters: Arbitrary key-value filters on custom metadata
            time_window: What a time phrase in `text_search` ("last week",
                "in March") does to relevance-sorted results. 'rank'
                (default): memories dated inside that period are ranked
                higher, and a strong match dated outside it is still
                returned. 'filter': only memories dated inside the period are
                returned, unless nothing at all falls inside it. Results
                sorted by 'recency', 'access_count' or 'strength' are always
                filtered.

        Returns:
            List of matching blocks.
        """
        if text_search and semantic and self._embedding_provider is not None:
            self._ensure_embedding_identity()
        return self._query.query(
            type=type,
            tags=tags,
            text_search=text_search,
            related_to=related_to,
            min_confidence=min_confidence,
            sort_by=sort_by,
            limit=limit,
            include_decayed=include_decayed,
            min_strength=min_strength,
            semantic=semantic,
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata_filters=metadata_filters,
            time_window=time_window,
        )

    def multi_hop_query(
        self,
        text_search: str,
        limit: int = 10,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[Block]:
        """
        Multi-hop iterative retrieval for complex reasoning queries.

        Performs multiple retrieval passes:
        1. Standard hybrid search for initial results
        2. Entity extraction + focused queries
        3. Graph walk from retrieved blocks with proximity boosting

        Best for questions requiring connecting multiple facts.
        """
        from memblock.multihop import MultiHopRetriever
        retriever = MultiHopRetriever(
            query_engine=self._query,
            graph=self._graph,
            storage=self._storage,
            decay=self._decay,
        )
        result = retriever.retrieve(
            query=text_search,
            limit=limit,
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata_filters=metadata_filters,
        )
        return result.blocks

    # ─── Context Builder ──────────────────────────────────────────────────

    def build_context(
        self,
        query: str | None = None,
        token_budget: int = 4000,
        strategy: str = "relevance",
        include_metadata: bool = True,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> str:
        """
        Build LLM-ready context from relevant memory blocks.

        Args:
            query: What to search for
            token_budget: Maximum tokens in output
            strategy: 'relevance', 'graph_walk', or 'type_grouped'
            include_metadata: Include confidence/source in output
            session_id: Filter by session (overrides default)
            org_id: Filter by organization (overrides default)
            project_id: Filter by project (overrides default)
            agent_id: Filter by agent (overrides default)
            metadata_filters: Arbitrary key-value filters on custom metadata

        Returns:
            Formatted string for LLM context injection.
        """
        return self._context.build_context(
            query=query,
            token_budget=token_budget,
            strategy=strategy,
            include_metadata=include_metadata,
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata_filters=metadata_filters,
        )

    # ─── Session Operations ────────────────────────────────────────────────

    def get_sessions(self) -> list[str]:
        """
        Get all distinct session IDs.

        Returns:
            List of session_id strings (excludes None).
        """
        blocks = self._storage.get_all_blocks()
        sessions: set[str] = set()
        for b in blocks:
            if b.metadata.session_id is not None:
                sessions.add(b.metadata.session_id)
        return sorted(sessions)

    def get_session_history(self, session_id: str, limit: int = 100) -> list[Block]:
        """
        Get blocks for a specific session, ordered by creation time.

        Args:
            session_id: The session to retrieve
            limit: Maximum blocks to return

        Returns:
            List of blocks in chronological order.
        """
        blocks = self._storage.query_blocks({
            "session_id": session_id,
            "sort_by": "created_at",
            "limit": limit,
        })
        # Sort chronologically (oldest first)
        blocks.sort(key=lambda b: b.metadata.created_at.isoformat())
        return blocks

    # ─── Auto-Extraction ─────────────────────────────────────────────────

    def extract(
        self,
        conversation: str,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> Any:
        """
        Auto-extract memory blocks from a conversation using an LLM.

        Args:
            conversation: The conversation text
            provider: "openai", "anthropic", or "custom"
            api_key: API key for the provider
            model: Model name (default: gpt-4o-mini for openai, claude-sonnet-4-20250514 for anthropic)
            base_url: Custom base URL (for OpenAI-compatible APIs)

        Returns:
            ExtractionResult with created block IDs.

        Requires: pip install memblock[llm]
        """
        from memblock.extraction import (
            LLMExtractor,
            OpenAIProvider,
            AnthropicProvider,
            ExtractionResult,
        )

        if api_key is None:
            raise ExtractionError("api_key is required for auto-extraction")

        try:
            if provider == "openai":
                llm_provider = OpenAIProvider(
                    api_key=api_key,
                    model=model or "gpt-4o-mini",
                    base_url=base_url,
                )
            elif provider == "anthropic":
                llm_provider = AnthropicProvider(
                    api_key=api_key,
                    model=model or "claude-sonnet-4-20250514",
                )
            else:
                raise ExtractionError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

            extractor = LLMExtractor(provider=llm_provider)
            return extractor.extract(conversation, memblock=self)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Extraction failed: {e}") from e

    def extract_messages(
        self,
        messages: list[dict[str, str]],
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
    ) -> Any:
        """
        Auto-extract from a list of message dicts.

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            provider: "openai" or "anthropic"
            api_key: API key
            model: Model name

        Returns:
            ExtractionResult
        """
        from memblock.extraction import (
            LLMExtractor,
            OpenAIProvider,
            AnthropicProvider,
        )

        if api_key is None:
            raise ExtractionError("api_key is required for auto-extraction")

        try:
            if provider == "openai":
                llm_provider = OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini")
            elif provider == "anthropic":
                llm_provider = AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514")
            else:
                raise ExtractionError(f"Unknown provider: {provider}")

            extractor = LLMExtractor(provider=llm_provider)
            return extractor.extract_from_messages(messages, memblock=self)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(f"Extraction failed: {e}") from e

    # ─── Background Extraction ──────────────────────────────────────────

    @property
    def extraction_pending(self) -> int:
        """Number of background extraction jobs still running or queued."""
        if self._bg_extractor is None:
            return 0
        return self._bg_extractor.pending_count

    @property
    def extraction_stats(self) -> dict[str, int]:
        """Background extraction worker statistics."""
        if self._bg_extractor is None:
            return {"pending": 0, "completed": 0, "failed": 0, "total_submitted": 0}
        return self._bg_extractor.stats

    def wait_for_extractions(self, timeout: float | None = None) -> None:
        """
        Block until all pending background extractions complete.

        Args:
            timeout: Max seconds to wait. None = wait forever.

        No-op if background extraction is not enabled.
        """
        if self._bg_extractor is not None:
            self._bg_extractor.wait(timeout=timeout)

    # ─── Opt-in Auto-Extraction ──────────────────────────────────────────

    def add_message(self, role: str, content: str) -> Any:
        """
        Buffer a message for auto-extraction.

        When auto_extract=True and message count reaches extract_every,
        extraction is triggered automatically. Returns ExtractionResult
        when extraction runs, None otherwise.

        On extraction failure, the buffer is preserved for retry via flush_extraction().
        """
        self._message_buffer.append({"role": role, "content": content})
        self._message_count += 1

        if (
            self._auto_extract
            and self._message_count % self._extract_every == 0
            and len(self._message_buffer) > 0
        ):
            return self._run_extraction()

        return None

    def flush_extraction(self) -> Any:
        """
        Manually trigger extraction on the current message buffer.

        On success: clears the buffer and returns ExtractionResult.
        On failure: keeps the buffer intact, returns ExtractionResult with errors.
        """
        if not self._message_buffer:
            from memblock.extraction import ExtractionResult
            return ExtractionResult(provider="none", errors=["No messages in buffer"])

        return self._run_extraction()

    def _classify_for_store(
        self, content, type, confidence, source, encryption_level, decay_rate, temporal_precision, tags,
    ):
        """Run the pluggable classifier and fill unset store() arguments from its decision.

        Returns the (possibly updated) arguments plus the audit record and a skip flag.
        Classifier failures never break store(): on any exception the arguments are returned
        unchanged and no record is written.
        """
        from memblock.classifier import CLASSIFIER_FIELDS, Classification, llm_classify

        try:
            result = Classification.from_result(self._classifier.classify(content))
        except Exception:
            return type, confidence, source, encryption_level, decay_rate, temporal_precision, tags, None, False
        if result is None:
            return type, confidence, source, encryption_level, decay_rate, temporal_precision, tags, None, False

        record: dict[str, Any] = {"classifier": getattr(self._classifier, "name", self._classifier.__class__.__name__),
                                  "escalated": result.escalate, "applied": {}, **result.record}
        if result.skip:
            record["skipped"] = True
            return type, confidence, source, encryption_level, decay_rate, temporal_precision, tags, record, True

        fields = {k: v for k, v in result.fields.items() if k in CLASSIFIER_FIELDS}
        extra = {k: v for k, v in result.fields.items() if k not in CLASSIFIER_FIELDS}
        if extra:
            record["extra"] = extra

        # Escalate to the LLM provider when the classifier is unsure and one is configured.
        if result.escalate and self._classifier_escalate and self._extract_provider_name is not None:
            try:
                override = llm_classify(self._init_extractor().provider, content)
                record["llm"] = override
                fields.update(override)
            except Exception as e:  # LLM failure: keep the classifier's answer
                record["llm_error"] = str(e)[:200]

        def _enum(cls, v):
            return v if isinstance(v, cls) else cls(v)

        if type is None and "type" in fields:
            type = _enum(BlockType, fields["type"]); record["applied"]["type"] = type.value
        if confidence is None and "confidence" in fields:
            confidence = float(fields["confidence"]); record["applied"]["confidence"] = confidence
        if source is None and "source" in fields:
            source = _enum(SourceType, fields["source"]); record["applied"]["source"] = source.value
        if encryption_level is None and "encryption_level" in fields:
            encryption_level = _enum(EncryptionLevel, fields["encryption_level"])
            record["applied"]["encryption_level"] = encryption_level.value
        if decay_rate is None and "decay_rate" in fields:
            decay_rate = float(fields["decay_rate"]); record["applied"]["decay_rate"] = decay_rate
        if temporal_precision == "exact" and "temporal_precision" in fields:
            temporal_precision = str(fields["temporal_precision"]); record["applied"]["temporal_precision"] = temporal_precision
        if tags is None and fields.get("tags"):
            tags = list(fields["tags"]); record["applied"]["tags"] = tags
        return type, confidence, source, encryption_level, decay_rate, temporal_precision, tags, record, False

    def _init_extractor(self) -> Any:
        """Lazily initialize the LLM extractor from configured params."""
        if self._extractor is not None:
            return self._extractor

        from memblock.extraction import (
            LLMExtractor,
            OpenAIProvider,
            AnthropicProvider,
            GeminiProvider,
            CallableProvider,
        )

        provider = self._extract_provider_name
        api_key = self._extract_api_key
        model = self._extract_model

        if provider is None:
            raise ExtractionError(
                "extract_provider is required when auto_extract=True. "
                "Use 'openai', 'anthropic', or 'gemini'."
            )

        if callable(provider):
            llm_provider = CallableProvider(fn=provider, provider_name="custom")
        elif provider == "openai":
            if not api_key:
                raise ExtractionError("extract_api_key is required for OpenAI extraction")
            llm_provider = OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini")
        elif provider == "anthropic":
            if not api_key:
                raise ExtractionError("extract_api_key is required for Anthropic extraction")
            llm_provider = AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514")
        elif provider == "gemini":
            if not api_key:
                raise ExtractionError("extract_api_key is required for Gemini extraction")
            llm_provider = GeminiProvider(api_key=api_key, model=model or "gemini-2.0-flash")
        else:
            raise ExtractionError(f"Unknown extract_provider: {provider}")

        self._extractor = LLMExtractor(
            provider=llm_provider,
            min_confidence=self._extract_min_confidence,
        )
        return self._extractor

    def _run_extraction(self) -> Any:
        """Run extraction on the current buffer. Clears buffer only on success."""
        from memblock.extraction import ExtractionResult

        try:
            extractor = self._init_extractor()
        except ExtractionError as e:
            return ExtractionResult(errors=[str(e)])

        try:
            result = extractor.extract_from_messages(
                self._message_buffer, memblock=self
            )
        except Exception as e:
            return ExtractionResult(errors=[f"Extraction failed: {e}"])

        # Only clear buffer on success (no errors or at least some blocks created)
        if not result.errors or result.blocks_created > 0:
            self._message_buffer = []

        return result

    # ─── Grouped Writes, Receipts ─────────────────────────────────────────

    def transaction(self, ttl: float = DEFAULT_LEASE_SECONDS) -> Transaction:
        """
        Group writes so they become visible together, or not at all.

            with mem.transaction():
                mem.store("User moved to Berlin")
                mem.store("User works at Acme")

        New blocks are written staged (stored, invisible to queries); changes to
        existing blocks are held back. Leaving the block publishes everything;
        an exception discards everything. Calling this while a transaction is
        already open on this thread joins that transaction.

        See `memblock.transactions` for how this stays all-or-nothing across a
        crash on every storage backend.
        """
        current = self._current_txn()
        if current is not None:
            return current
        return Transaction(self, ttl=ttl)

    def recover_transactions(self, force: bool = False) -> dict[str, int]:
        """
        Finish or discard transactions a crash left behind.

        Runs by itself the first time this instance opens a transaction. A
        transaction that reached its commit point is completed; one that did not
        is discarded once its lease has expired (`force=True` discards it now).
        """
        return recover_transactions(self, force=force)

    def receipt_status(self, receipt: str) -> ReceiptStatus:
        """Where the write stored under this receipt stands: applied (visible),
        pending (staged in an open transaction) or unknown (nothing stored)."""
        found = self._storage.query_blocks(
            {"deleted": True, "metadata_filters": {RECEIPT_KEY: receipt}, "limit": 5}
        )
        found = [b for b in found if (b.metadata.custom_metadata or {}).get(RECEIPT_KEY) == receipt]
        if any(not b.deleted for b in found):
            return ReceiptStatus.APPLIED
        if any(is_staged(b) for b in found):
            return ReceiptStatus.PENDING
        return ReceiptStatus.UNKNOWN

    def _current_txn(self) -> Transaction | None:
        return getattr(self._txn_local, "txn", None)

    def _txn_enter(self, txn: Transaction) -> None:
        if self._current_txn() is not None:
            raise TransactionError("A transaction is already open on this thread")
        self._txn_local.txn = txn

    def _txn_exit(self, txn: Transaction) -> None:
        if self._current_txn() is txn:
            self._txn_local.txn = None

    def _recover_once(self) -> None:
        if self._txn_recovered:
            return
        self._txn_recovered = True
        try:
            recover_transactions(self)
        except Exception:
            pass  # recovery can be retried; it must not block new work

    def _plaintext(self, block: Block) -> str | None:
        """A block's readable content, or None when it is sealed and this
        instance cannot open it."""
        if not block.encrypted:
            return block.content
        if not self._crypto.enabled:
            return None
        try:
            return self._crypto.open(block.content, block.encryption_level)
        except Exception:
            return None

    def _reveal(self, block: Block) -> Block:
        """A copy of the block with readable content where this instance can open it."""
        text = self._plaintext(block)
        if text is None or text == block.content:
            return block
        shown = Block.from_dict(block.to_dict())
        shown.content = text
        return shown

    def _receipt_replay(self, receipt: str, txn: Transaction | None) -> Block | None:
        """The block already written under this receipt, if there is one."""
        if not isinstance(receipt, str) or not receipt:
            raise ValidationError("receipt must be a non-empty string")
        if txn is not None and txn.receipts.get(receipt) == ReceiptStatus.APPLIED:
            for block_id in txn._staged:
                staged = self._storage.get_block(block_id)
                if staged is not None and (staged.metadata.custom_metadata or {}).get(RECEIPT_KEY) == receipt:
                    return self._reveal(staged)
        found = self._storage.query_blocks({"metadata_filters": {RECEIPT_KEY: receipt}, "limit": 1})
        for block in found:
            if (block.metadata.custom_metadata or {}).get(RECEIPT_KEY) == receipt:
                return self._reveal(block)
        return None

    def _create_block(
        self,
        content: str,
        *,
        type: BlockType,
        confidence: float,
        source: SourceType,
        tags: list[str] | None,
        parent_id: str | None,
        encryption_level: EncryptionLevel,
        decay_rate: float,
        ttl: int | None,
        session_id: str | None,
        org_id: str | None,
        project_id: str | None,
        agent_id: str | None,
        metadata: dict[str, Any] | None,
        happened_at: Any | None,
        happened_at_end: Any | None,
        temporal_precision: str,
        receipt: str | None,
        txn: Transaction | None,
    ) -> Block:
        """Encrypt, persist and embed one block. Inside a transaction the block
        is written staged and published with the rest."""
        if self._embedding_provider is not None:
            self._ensure_embedding_identity()

        custom = dict(metadata) if metadata else None
        if receipt is not None or txn is not None:
            custom = custom or {}
            if receipt is not None:
                custom[RECEIPT_KEY] = receipt
            if txn is not None:
                custom[TXN_KEY] = txn.id
        if txn is not None and receipt is not None:
            txn._receipt(receipt, ReceiptStatus.PENDING)

        try:
            # Encrypt if needed
            stored_content = content
            encrypted = False
            if encryption_level != EncryptionLevel.NONE and self._crypto.enabled:
                try:
                    stored_content = self._crypto.seal(content, encryption_level)
                except Exception as e:
                    raise EncryptionError(f"Failed to encrypt block content: {e}") from e
                encrypted = True

            block = self._store.create(
                content=stored_content,
                type=type,
                confidence=confidence,
                source=source,
                tags=tags,
                parent_id=parent_id,
                encryption_level=encryption_level,
                decay_rate=decay_rate,
                ttl=ttl,
                session_id=session_id,
                org_id=org_id,
                project_id=project_id,
                agent_id=agent_id,
                custom_metadata=custom,
                happened_at=happened_at,
                happened_at_end=happened_at_end,
                temporal_precision=temporal_precision,
                deleted=txn is not None,
            )

            if encrypted:
                self._storage.update_block(block.id, {"encrypted": True})
                block.encrypted = True

            # Generate and store embedding (if provider available)
            # Use original plaintext content for embedding, not encrypted content
            if self._embedding_provider is not None:
                self._embed_block(block.id, content)
        except BaseException:
            if txn is not None and receipt is not None:
                txn._receipt(receipt, ReceiptStatus.ABORTED)
            raise

        if txn is not None:
            txn._stage(block.id)
            if receipt is not None:
                txn._receipt(receipt, ReceiptStatus.APPLIED)
        return block

    def _defer_update(self, txn: Transaction, block_id: str, updates: dict[str, Any]) -> Block | None:
        block = self._store.get_without_touch(block_id)
        if block is None:
            return None
        plain = updates.get("content")
        held: dict[str, Any] = {}
        for key, value in updates.items():
            if key == "content" and block.encrypted and self._crypto.enabled:
                value = self._crypto.seal(value, block.encryption_level)
            held[key] = value.value if isinstance(value, Enum) else value
        txn._defer({"op": "update", "block_id": block_id, "updates": held})

        def _after() -> None:
            # A sealed block's new vector needs the plain text, which only this
            # process has; an unsealed one is re-embedded when the change is applied.
            if plain is not None and block.encrypted and self._embedding_provider is not None:
                self._embed_block(block_id, plain)
            fresh = self._store.get_without_touch(block_id)
            if fresh is not None:
                self._hooks.emit(EventType.ON_UPDATE, {
                    "block_id": block_id,
                    "block": fresh,
                    "updates": updates,
                })

        txn._on_publish(_after)
        return block

    def _defer_delete(self, txn: Transaction, block_id: str, cascade: bool) -> bool:
        block = self._storage.get_block(block_id)
        if block is None:
            return False
        if block_id in txn._staged:
            # Written by this transaction and seen by nobody else: drop it now.
            self._storage.delete_embedding(block_id)
            self._storage.delete_edges_for_block(block_id)
            self._storage.delete_block(block_id)
            txn._staged.remove(block_id)
            return True
        txn._defer({"op": "delete", "block_id": block_id, "cascade": cascade})
        txn._on_publish(lambda: self._hooks.emit(EventType.ON_DELETE, {
            "block_id": block_id,
            "cascade": cascade,
        }))
        return True

    def _defer_link(
        self,
        txn: Transaction,
        source_id: str,
        target_id: str,
        relation: EdgeRelation,
        weight: float,
    ) -> None:
        # Check now what the graph would check on publish, so a bad link is
        # refused while the transaction can still be aborted.
        source = self._storage.get_block(source_id)
        target = self._storage.get_block(target_id)
        if source is None:
            raise ValueError(f"Source block {source_id} not found")
        if target is None:
            raise ValueError(f"Target block {target_id} not found")
        BlockSchema.validate_edge(relation, source.type, target.type)
        txn._defer({
            "op": "link",
            "source_id": source_id,
            "target_id": target_id,
            "relation": relation.value,
            "weight": weight,
        })

    # ─── Verbatim Episodes ────────────────────────────────────────────────

    def store_episode(
        self,
        content: str,
        role: str = "user",
        session_id: str | None = None,
        turn: int | None = None,
        seq: int | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        happened_at: Any | None = None,
        encryption_level: EncryptionLevel | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        receipt: str | None = None,
    ) -> Block:
        """
        Store one message or document chunk word for word.

        An episode is the record of what was actually said, so it skips
        everything that could change or drop it: the classifier, conflict
        resolution, deduplication and extraction. It never decays, so `prune()`
        never removes it. Facts extracted from it should link back with
        `EdgeRelation.DERIVED_FROM`.

        Args:
            content: The exact text.
            role: Who said it: "user", "assistant", "tool", "system", "document".
            session_id: The conversation it belongs to.
            turn: The turn number within the session.
            seq: The message's position within the turn.
            receipt: See `store()`.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValidationError("An episode needs non-empty content")
        txn = self._current_txn()
        if receipt is not None:
            replay = self._receipt_replay(receipt, txn)
            if replay is not None:
                return replay

        meta = dict(metadata or {})
        meta["_role"] = role
        if turn is not None:
            meta["_turn"] = turn
        if seq is not None:
            meta["_seq"] = seq

        block = self._create_block(
            content,
            type=BlockType.EPISODE,
            confidence=1.0,
            source=SourceType.EXPLICIT if role in ("user", "document") else SourceType.OBSERVED,
            tags=tags,
            parent_id=None,
            encryption_level=encryption_level or EncryptionLevel.NONE,
            decay_rate=0.0,
            ttl=None,
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata=meta,
            happened_at=happened_at if happened_at is not None else now_utc(),
            happened_at_end=None,
            temporal_precision="exact",
            receipt=receipt,
            txn=txn,
        )

        def _announce() -> None:
            self._hooks.emit(EventType.ON_ADD, {
                "block_id": block.id,
                "block": block,
                "content": content,
                "type": BlockType.EPISODE.value,
            })

        if txn is not None:
            txn._on_publish(_announce)
        else:
            _announce()
        return block

    # ─── Snapshots ────────────────────────────────────────────────────────

    def snapshot(
        self,
        path: str,
        session_id: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Save this memory to one file: the whole store, or one session's blocks.

        The file carries a stamp (format, checksum, embedding model). Returns
        the stamp with `path` added. See `memblock.snapshot`.
        """
        from memblock.snapshot import write_snapshot
        return write_snapshot(self, path, session_id=session_id, overwrite=overwrite)

    def restore(
        self,
        path: str,
        mode: str = "empty",
        embeddings: str = "strict",
    ) -> dict[str, Any]:
        """
        Load a snapshot into this store.

        mode="empty" (default) needs a store with no blocks and keeps the
        original history; mode="merge" adds to what is here. A snapshot whose
        checksum fails, or whose vectors were made by a different embedding
        model, is refused (`embeddings="drop"` or `"reembed"` takes the blocks
        without the foreign vectors). See `memblock.snapshot`.
        """
        from memblock.snapshot import restore_snapshot
        return restore_snapshot(self, path, mode=mode, embeddings=embeddings)

    def fork(self, storage: str, **kwargs: Any) -> "MemBlock":
        """
        Copy this memory into a new, independent store and return it.

        `storage` is the new store's URI and must be empty. Other keyword
        arguments go to the new `MemBlock`; the embedding provider is carried
        over unless you pass one. Sealed content is copied sealed, so pass the
        same `encryption_key` to read it in the copy.
        """
        import os
        import tempfile

        kwargs.setdefault("embeddings", self._embedding_provider or False)
        kwargs.setdefault("user_id", self._user_id)
        descriptor, path = tempfile.mkstemp(prefix="memblock-fork-", suffix=".json")
        os.close(descriptor)
        try:
            self.snapshot(path, overwrite=True)
            copy = MemBlock(storage=storage, **kwargs)
            try:
                copy.restore(path)
            except BaseException:
                copy.close()
                raise
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return copy

    def embedding_identity(self) -> str | None:
        """The embedding model this instance is configured with, as recorded in
        snapshot stamps and in the store. None when embeddings are off."""
        from memblock.snapshot import embedding_identity
        return embedding_identity(self._embedding_provider)

    def reembed(self) -> int:
        """
        Rebuild every vector with this instance's embedding model and record
        that model as the store's. Use after changing embedding models (open
        the store with `on_embedding_mismatch="ignore"`). Returns the number of
        blocks embedded; sealed blocks this instance cannot open are skipped.
        """
        if self._embedding_provider is None:
            return 0
        from memblock.snapshot import embedding_identity, write_manifest
        count = 0
        for block in self._storage.get_all_blocks():
            if is_internal(block):
                continue
            text = self._plaintext(block)
            if text is None:
                continue
            self._embed_block(block.id, text)
            count += 1
        write_manifest(self, embedding_identity(self._embedding_provider))
        self._embedding_identity_checked = True
        return count

    def _ensure_embedding_identity(self) -> None:
        if self._embedding_identity_checked:
            return
        from memblock.snapshot import check_embedding_identity
        check_embedding_identity(self, self._on_embedding_mismatch)
        self._embedding_identity_checked = True

    # ─── Integrity ────────────────────────────────────────────────────────

    def verify(self) -> TamperReport:
        """
        Verify the integrity of the operation log hash chain.

        Returns a TamperReport indicating if any tampering was detected.
        """
        return self._store.op_log.verify()

    # ─── Decay & Maintenance ──────────────────────────────────────────────

    def prune(self, min_strength: float = 0.1) -> list[Block]:
        """Remove decayed memories below the strength threshold."""
        return self._decay.prune(min_strength=min_strength)

    def strongest(self, limit: int = 10) -> list[tuple[Block, float]]:
        """Get the strongest memories."""
        return self._decay.get_strongest(limit=limit)

    def weakest(self, limit: int = 10) -> list[tuple[Block, float]]:
        """Get the weakest memories (candidates for pruning)."""
        return self._decay.get_weakest(limit=limit)

    # ─── Embeddings ──────────────────────────────────────────────────────

    @property
    def has_embeddings(self) -> bool:
        """Whether embedding-based semantic search is enabled."""
        return self._embedding_provider is not None

    def _embed_block(self, block_id: str, content: str) -> None:
        """Generate and store an embedding for a block."""
        if self._embedding_provider is None:
            return
        try:
            from memblock.embeddings import pack_embedding
            vectors = self._embedding_provider.embed([content])
            if vectors:
                self._storage.save_embedding(block_id, pack_embedding(vectors[0]))
        except Exception:
            pass  # Embedding failure should not break store operations

    # ─── Export ───────────────────────────────────────────────────────────

    def export_markdown(self) -> str:
        """Export all memories as human-readable markdown."""
        blocks = self._storage.get_all_blocks()
        lines = ["# MemBlock Export", ""]

        for block in blocks:
            lines.append(f"## [{block.type.value.upper()}] {block.content[:80]}")
            lines.append(f"- **ID**: {block.id}")
            lines.append(f"- **Confidence**: {block.metadata.confidence:.2f}")
            lines.append(f"- **Source**: {block.metadata.source.value}")
            lines.append(f"- **Created**: {block.metadata.created_at.isoformat()}")
            lines.append(f"- **Access Count**: {block.metadata.access_count}")
            lines.append(f"- **Tags**: {', '.join(block.tags) if block.tags else 'none'}")

            edges = self._storage.get_edges(block.id)
            if edges:
                lines.append(f"- **Edges**:")
                for edge in edges:
                    direction = "→" if edge.source_id == block.id else "←"
                    other_id = edge.target_id if edge.source_id == block.id else edge.source_id
                    lines.append(f"  - {direction} {edge.relation.value} {other_id} (w={edge.weight})")

            lines.append("")

        return "\n".join(lines)

    # ─── Analytics ────────────────────────────────────────────────────────

    def _get_analytics(self) -> Any:
        """Lazily initialize the OrgAnalytics engine."""
        if self._analytics is not None:
            return self._analytics

        from memblock.analytics import OrgAnalytics, NoiseFilter

        noise = NoiseFilter(custom_noise=self._analytics_noise_words)
        self._storage.initialize_analytics_tables()
        self._analytics = OrgAnalytics(self._storage, noise_filter=noise)
        return self._analytics

    def log_question(
        self,
        question: str,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> Any:
        """Log a user question for org-level analytics.

        Requires enable_analytics=True. Returns QuestionRecord or None (if noise-filtered).
        """
        if not self._enable_analytics:
            raise AnalyticsError("Analytics not enabled. Pass enable_analytics=True to MemBlock().")
        return self._get_analytics().log_question(
            org_id=org_id or self._org_id or "default",
            user_id=user_id or self._user_id,
            question=question,
        )

    def get_top_questions(
        self,
        org_id: str | None = None,
        limit: int = 20,
        since: Any = None,
        until: Any = None,
    ) -> list:
        """Get most frequently asked questions at the org level."""
        if not self._enable_analytics:
            raise AnalyticsError("Analytics not enabled. Pass enable_analytics=True to MemBlock().")
        return self._get_analytics().get_top_questions(
            org_id=org_id or self._org_id or "default",
            limit=limit,
            since=since,
            until=until,
        )

    def get_trending_questions(
        self,
        org_id: str | None = None,
        window_days: int = 7,
        limit: int = 10,
    ) -> list:
        """Get questions trending upward in frequency."""
        if not self._enable_analytics:
            raise AnalyticsError("Analytics not enabled. Pass enable_analytics=True to MemBlock().")
        return self._get_analytics().get_trending(
            org_id=org_id or self._org_id or "default",
            window_days=window_days,
            limit=limit,
        )

    def get_question_breakdown(
        self,
        question: str,
        org_id: str | None = None,
        granularity: str = "daily",
    ) -> dict:
        """Get daily/weekly/monthly frequency breakdown for a question."""
        if not self._enable_analytics:
            raise AnalyticsError("Analytics not enabled. Pass enable_analytics=True to MemBlock().")
        return self._get_analytics().get_question_breakdown(
            org_id=org_id or self._org_id or "default",
            question_text=question,
            granularity=granularity,
        )

    def question_stats(self, org_id: str | None = None) -> dict:
        """Get summary analytics stats for an org."""
        if not self._enable_analytics:
            raise AnalyticsError("Analytics not enabled. Pass enable_analytics=True to MemBlock().")
        return self._get_analytics().question_stats(
            org_id=org_id or self._org_id or "default",
        )

    # ─── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get statistics about the memory store."""
        blocks = self._storage.get_all_blocks()
        all_blocks = [
            b for b in self._storage.get_all_blocks(include_deleted=True)
            if not is_internal(b)
        ]

        type_counts: dict[str, int] = {}
        for b in blocks:
            key = b.type.value
            type_counts[key] = type_counts.get(key, 0) + 1

        total_edges = 0
        for b in blocks:
            total_edges += len(self._storage.get_edges(b.id, direction="outgoing"))

        embedding_count = len(self._storage.get_all_embeddings())

        return {
            "total_blocks": len(blocks),
            "deleted_blocks": len(all_blocks) - len(blocks),
            "blocks_by_type": type_counts,
            "total_edges": total_edges,
            "total_operations": len(self._storage.get_operations()),
            "embeddings_enabled": self.has_embeddings,
            "total_embeddings": embedding_count,
        }

    # ─── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the storage connection and shut down background workers."""
        if self._bg_extractor is not None:
            self._bg_extractor.shutdown(wait=True)
        self._storage.close()

    def __enter__(self) -> MemBlock:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self.stats()
        emb = " +embeddings" if self.has_embeddings else ""
        return f"MemBlock(blocks={stats['total_blocks']}, edges={stats['total_edges']}{emb})"
