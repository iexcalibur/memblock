"""AsyncMemBlock — async-native MemBlock for asyncio frameworks.

Two modes, picked from the storage URL:

  - **Legacy mode** (sqlite://, postgresql://, anything not starting
    with `postgresql+asyncpg://`):
        Constructs a sync `MemBlock` internally and wraps every I/O
        method with `asyncio.to_thread()`. Exists for backward compat
        and for SQLite / non-async storage backends.

  - **Native-async mode** (postgresql+asyncpg://):
        Bypasses the sync `MemBlock` entirely. Wires the
        `AsyncPostgreSQLAdapter` directly into native-async
        equivalents of `QueryEngine` / `ContextBuilder` /
        `ConflictResolver` / `OpLog` / auto-link / embedding-gen.
        No `asyncio.to_thread` overhead on storage I/O.

Mode selection is automatic — pass a `postgresql+asyncpg://...`
URL and the native path activates.

Usage:
    async with AsyncMemBlock(
        storage="postgresql+asyncpg://user:pass@host/db",
        user_id="u_123",
        embeddings="gemini",
        embeddings_api_key="...",
    ) as mem:
        block = await mem.store("...", type=BlockType.PREFERENCE)
        results = await mem.query(text_search="...", semantic=True)
        ctx = await mem.build_context(query="...")

The legacy `.sync` property still works in legacy mode but raises
`RuntimeError` in native mode — methods that previously needed
`.sync` (`add_message`, `enable_auto_link`, `flush_extraction`)
now have first-class async equivalents on `AsyncMemBlock` itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from typing import Any

from memblock.block import Block
from memblock.dedup import (
    ContentHasher,
    DuplicateChecker,
    DuplicatePolicy,
)
from memblock.hooks import EventType, HookManager
from memblock.errors import (
    DuplicateBlockError,
    EncryptionError,
)
from memblock.memblock import MemBlock
from memblock.ops import TamperReport
from memblock.types import (
    BlockMetadata,
    BlockType,
    EdgeRelation,
    EncryptionLevel,
    Edge,
    Operation,
    OpAction,
    SourceType,
    generate_block_id,
    generate_edge_id,
    generate_op_id,
    now_utc,
)


# ── Native-mode URL detection ────────────────────────────────────────

_NATIVE_ASYNC_URL_RE = re.compile(r"^postgresql\+asyncpg://")


def _is_native_async_url(storage: str | None) -> bool:
    """True when the URL scheme picks the native asyncpg adapter."""
    if not isinstance(storage, str):
        return False
    return bool(_NATIVE_ASYNC_URL_RE.match(storage))


# ── Entity-extraction helpers (multi_hop_query) ─────────────────────
# Pure-Python regex extraction. Mirrors `MultiHopRetriever`'s logic so
# native mode produces equivalent multi-hop results.

_MULTIHOP_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "has", "his", "how",
    "its", "may", "new", "now", "old", "see", "way", "who", "did",
    "get", "let", "say", "she", "too", "use", "yes", "yet", "hey",
    "said", "also", "been", "call", "come", "each", "from", "have",
    "here", "just", "like", "made", "make", "much", "must", "name",
    "only", "over", "such", "take", "than", "that", "them", "then",
    "they", "this", "very", "when", "what", "with", "will", "well",
    "your", "about", "after", "could", "every", "first", "found",
    "great", "house", "large", "later", "never", "other", "place",
    "right", "shall", "small", "some", "still", "their", "there",
    "these", "thing", "think", "those", "three", "under", "where",
    "which", "while", "world", "would", "yeah", "okay", "sure",
    "really", "actually", "session", "user", "speaker", "unknown",
})


def _extract_entities_from_text(text: str) -> list[str]:
    """Lift capitalized phrases + multi-word noun phrases as entity
    candidates. Mirrors `MultiHopRetriever._extract_entities_from_text`.
    """
    entities: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        lower = t.strip().lower()
        if (lower not in seen
                and len(t.strip()) > 2
                and lower not in _MULTIHOP_STOP_WORDS):
            seen.add(lower)
            entities.append(t.strip())

    # Capitalized phrases (Proper Nouns)
    for match in re.findall(r"\b[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+)*", text):
        _add(match)

    # Lower-case multi-word noun phrases (3+ chars per word)
    for match in re.findall(r"\b[a-z]{3,}(?:\s+[a-z]{3,}){0,2}\b", text):
        _add(match)

    return entities[:20]


def _extract_entities_from_blocks(blocks: list) -> list[str]:
    """Same regex extraction over a list of blocks' content."""
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        for ent in _extract_entities_from_text(block.content):
            lower = ent.lower()
            if lower not in seen:
                seen.add(lower)
                out.append(ent)
    return out


class AsyncMemBlock:
    """Async-compatible MemBlock with optional native-async path.

    See module docstring for the two modes and URL-based selection.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Same constructor surface as `MemBlock`. URL routing is
        transparent — pass `postgresql+asyncpg://...` to opt into
        native async."""
        storage_url = kwargs.get("storage")
        self._native_async = _is_native_async_url(storage_url)

        if self._native_async:
            self._init_native(**kwargs)
        else:
            # Legacy path: thin wrapper over sync MemBlock.
            self._mem: MemBlock | None = MemBlock(**kwargs)
            self._init_native_state_stubs()

    # ─── Native-mode init ────────────────────────────────────────────

    def _init_native_state_stubs(self) -> None:
        """Initialize attributes accessed from both modes to safe
        default values, even when in legacy mode (so we can
        gracefully share helpers without `hasattr` checks)."""
        # Native-mode pieces — None / empty in legacy mode
        self._async_storage: Any = None
        self._async_query: Any = None
        self._async_context: Any = None
        self._async_conflict_resolver: Any = None
        self._async_extractor: Any = None
        self._embedding_provider: Any = None
        self._dedup: DuplicateChecker | None = None
        self._on_duplicate: DuplicatePolicy | None = None
        self._crypto: Any = None
        self._user_id: str = "default"
        self._session_id: str | None = None
        self._org_id: str | None = None
        self._project_id: str | None = None
        self._agent_id: str | None = None
        self._author: str = "agent"

        # Op-log state — hashes + clock initialized lazily on first
        # store/update/delete in native mode.
        self._op_clock: int | None = None
        self._last_op_hash: str = ""

        # Auto-link state
        self._auto_link_enabled: bool = False
        self._auto_link_max_neighbors: int = 5
        self._last_stored_id: str | None = None
        self._tag_index: dict[str, list[str]] = {}

        # Auto-extract / message buffer
        self._auto_extract: bool = False
        self._auto_extract_on_store: bool = False
        self._background_extract: bool = False
        self._extract_every: int = 100
        self._extract_min_confidence: float = 0.3
        self._message_buffer: list[dict[str, str]] = []
        self._message_count: int = 0

        # Background-task bookkeeping for `auto_extract_on_store +
        # background_extract`. Tasks are kept in a strong-ref set so
        # they don't get GC'd before completion; `wait_for_extractions`
        # awaits them all.
        self._bg_tasks: set[asyncio.Task] = set()

        # Conflict resolution gate
        self._conflict_resolution: bool = False
        self._extracting: bool = False  # recursion guard

        # Hook manager — fires lifecycle events from native I/O paths.
        # Sync MemBlock owns one too; we instantiate our own so native
        # mode emits without going through the sync layer.
        self._hooks: HookManager = HookManager()

    def _init_native(self, **kwargs: Any) -> None:
        """Wire up native-async storage + smart-layer pieces."""
        from memblock.async_query import AsyncQueryEngine
        from memblock.async_context import AsyncContextBuilder
        from memblock.storage.async_postgresql import AsyncPostgreSQLAdapter
        from memblock.crypto import CryptoLayerWithPassphrase

        self._native_async = True
        self._mem = None  # No sync MemBlock in native mode

        # First reset all stub attributes so we have somewhere to
        # land config values.
        self._init_native_state_stubs()

        storage_url = kwargs["storage"]
        self._user_id = str(kwargs.get("user_id", "default"))
        self._session_id = kwargs.get("session_id")
        self._org_id = kwargs.get("org_id")
        self._project_id = kwargs.get("project_id")
        self._agent_id = kwargs.get("agent_id")
        self._author = str(kwargs.get("author", "agent"))

        # ── Storage
        self._async_storage = AsyncPostgreSQLAdapter(
            dsn=storage_url,
            user_id=self._user_id,
            schema=kwargs.get("schema", "public"),
            pool=kwargs.get("pool"),
            pool_min_size=kwargs.get("pool_min_size", 2),
            pool_max_size=kwargs.get("pool_max_size", 20),
        )

        # ── Embedding provider (sync; HTTP-bound — wrapped via to_thread)
        embeddings = kwargs.get("embeddings", False)
        if embeddings:
            self._embedding_provider = MemBlock._create_embedding_provider(
                embeddings,
                kwargs.get("embeddings_api_key"),
                kwargs.get("embeddings_model"),
            )

        # ── Deduplicator
        on_duplicate_raw = kwargs.get("on_duplicate")
        similarity_threshold = kwargs.get("similarity_threshold", 0.95)
        if on_duplicate_raw is not None:
            from memblock.dedup import DuplicatePolicy as _DP
            self._on_duplicate = (
                _DP(on_duplicate_raw)
                if isinstance(on_duplicate_raw, str)
                else on_duplicate_raw
            )
            # Deduplicator uses storage; we'll wire the async adapter
            # for content-hash lookups on demand inside `store()`.
            self._dedup = None  # native dedup is inlined

        # ── Crypto
        encryption_key = kwargs.get("encryption_key")
        if encryption_key is not None:
            self._crypto = CryptoLayerWithPassphrase(passphrase=encryption_key)
        else:
            self._crypto = None

        # ── Async query + context (lazy — both need storage initialized).
        # Forward `reranker=` from kwargs so callers can pass any
        # `Reranker` subclass (BM25Reranker / CohereReranker /
        # CrossEncoderReranker / HeuristicReranker / CallableReranker).
        # The query engine applies it best-effort after FTS+vector
        # ranking when text_search is non-empty.
        self._async_query = AsyncQueryEngine(
            storage=self._async_storage,
            embedding_provider=self._embedding_provider,
            reranker=kwargs.get("reranker"),
        )
        self._reranker = kwargs.get("reranker")
        self._async_context = AsyncContextBuilder(
            storage=self._async_storage,
            query_engine=self._async_query,
        )

        # ── Conflict resolution config
        self._conflict_resolution = bool(kwargs.get("conflict_resolution", False))
        self._extract_provider_name = kwargs.get("extract_provider")
        self._extract_api_key = kwargs.get("extract_api_key")
        self._extract_model = kwargs.get("extract_model")

        # ── Auto-extract config
        self._auto_extract = bool(kwargs.get("auto_extract", False))
        self._auto_extract_on_store = bool(
            kwargs.get("auto_extract_on_store", False),
        )
        self._background_extract = bool(kwargs.get("background_extract", False))
        self._extract_every = int(kwargs.get("extract_every", 100))
        self._extract_min_confidence = float(
            kwargs.get("extract_min_confidence", 0.3),
        )

        # Pre-register any hook callbacks the user passed at construction
        # time (mirrors sync MemBlock's `hooks={"on_add": [cb,...]}` kwarg).
        for event_name, callbacks in (kwargs.get("hooks") or {}).items():
            for callback in (callbacks or []):
                self._hooks.register(event_name, callback)

        # ── Schema bootstrap is deferred to the first awaited method.
        # `initialize()` is async; we can't run it from a sync
        # __init__. Tracked via a flag so we run it once.
        self._initialized = False
        self._init_lock = asyncio.Lock()

    # ─── Mode helpers ─────────────────────────────────────────────────

    @property
    def is_native_async(self) -> bool:
        """True when this instance routes I/O through asyncpg
        natively. False when it wraps a sync MemBlock."""
        return self._native_async

    @property
    def sync(self) -> MemBlock:
        """The underlying sync MemBlock — only present in legacy mode.

        In native-async mode, raises `RuntimeError` with a migration
        hint pointing the caller at the native async equivalents
        (`add_message`, `enable_auto_link`, `flush_extraction`, etc.)
        which are now first-class on AsyncMemBlock itself."""
        if self._native_async:
            raise RuntimeError(
                "AsyncMemBlock.sync is unavailable in native-async mode "
                "(postgresql+asyncpg://...). The legacy `.sync.<method>()` "
                "calls have first-class async equivalents on AsyncMemBlock — "
                "use `await mem.<method>(...)` instead."
            )
        assert self._mem is not None
        return self._mem

    async def _ensure_initialized(self) -> None:
        """Lazy schema initialization for native-async mode. Runs
        once per instance; concurrent first-callers serialize via
        the init lock."""
        if not self._native_async:
            return
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._async_storage.initialize()
            self._initialized = True

    # ─── Store Operations ─────────────────────────────────────────────

    async def store(
        self,
        content: str,
        type: BlockType = BlockType.FACT,
        confidence: float = 1.0,
        source: SourceType = SourceType.EXPLICIT,
        tags: list[str] | None = None,
        parent_id: str | None = None,
        encryption_level: EncryptionLevel = EncryptionLevel.NONE,
        decay_rate: float = 0.01,
        ttl: int | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        happened_at: Any | None = None,
        happened_at_end: Any | None = None,
        temporal_precision: str = "exact",
    ) -> Block | None:
        """Store a new memory block.

        In native-async mode: runs conflict resolution (if enabled),
        dedup check, encryption, persistent block + metadata write,
        op-log append, embedding generation, auto-link edge writes,
        and auto-extract-on-store — all natively async, no
        `asyncio.to_thread` on storage I/O.

        In legacy mode: delegates to `MemBlock.store` via to_thread.
        """
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.store,
                content=content, type=type, confidence=confidence,
                source=source, tags=tags, parent_id=parent_id,
                encryption_level=encryption_level, decay_rate=decay_rate,
                ttl=ttl, session_id=session_id, org_id=org_id,
                project_id=project_id, agent_id=agent_id,
                metadata=metadata, happened_at=happened_at,
                happened_at_end=happened_at_end,
                temporal_precision=temporal_precision,
            )

        await self._ensure_initialized()

        # ── Conflict resolution (LLM-driven)
        if (
            self._conflict_resolution
            and not self._extracting
            and self._embedding_provider is not None
            and self._extract_provider_name is not None
        ):
            try:
                similar = await self.query(
                    text_search=content, semantic=True, limit=5,
                    session_id=session_id or self._session_id,
                )
                if similar:
                    from memblock.conflict import (
                        AsyncConflictResolver, ConflictActionType,
                    )
                    if self._async_conflict_resolver is None:
                        provider = await self._init_extractor_provider()
                        self._async_conflict_resolver = AsyncConflictResolver(
                            provider=provider,
                        )
                    result = await self._async_conflict_resolver.aresolve(
                        content, similar,
                    )
                    for action in result.actions:
                        if (action.action == ConflictActionType.UPDATE
                                and action.block_id):
                            await self.update(
                                action.block_id,
                                content=action.new_content or content,
                            )
                            return await self.get(action.block_id)
                        if (action.action == ConflictActionType.DELETE
                                and action.block_id):
                            await self.delete(action.block_id)
                        elif action.action == ConflictActionType.NONE:
                            return None
                        # ADD falls through to normal store
            except Exception:
                pass  # CR failure → normal store path

        # ── Dedup check
        if self._on_duplicate is not None:
            content_hash = ContentHasher.hash(content)
            existing = await self._async_storage.get_block_by_content_hash(
                content_hash,
            )
            if existing is not None:
                if self._on_duplicate == DuplicatePolicy.ERROR:
                    raise DuplicateBlockError(
                        f"Duplicate content detected (block {existing.id})"
                    )
                if self._on_duplicate == DuplicatePolicy.SKIP:
                    return None
                if self._on_duplicate == DuplicatePolicy.RETURN_EXISTING:
                    return existing
                if self._on_duplicate == DuplicatePolicy.MERGE:
                    merged_tags = list(set(existing.tags + (tags or [])))
                    merged_confidence = max(
                        existing.metadata.confidence, confidence,
                    )
                    await self.update(
                        existing.id,
                        tags=merged_tags,
                        confidence=merged_confidence,
                    )
                    existing.tags = merged_tags
                    existing.metadata.confidence = merged_confidence
                    return existing

        # ── Encryption
        stored_content = content
        encrypted_flag = False
        if (
            encryption_level != EncryptionLevel.NONE
            and self._crypto is not None
            and getattr(self._crypto, "enabled", False)
        ):
            try:
                stored_content = self._crypto.seal(content, encryption_level)
            except Exception as e:
                raise EncryptionError(
                    f"Failed to encrypt block content: {e}"
                ) from e
            encrypted_flag = True

        # ── Build block
        block = Block(
            id=generate_block_id(),
            type=type,
            content=stored_content,
            metadata=BlockMetadata(
                confidence=confidence,
                source=source,
                created_at=now_utc(),
                created_by=self._author,
                decay_rate=decay_rate,
                ttl=ttl,
                session_id=session_id or self._session_id,
                org_id=org_id or self._org_id,
                project_id=project_id or self._project_id,
                agent_id=agent_id or self._agent_id,
                custom_metadata=metadata,
                happened_at=happened_at,
                happened_at_end=happened_at_end,
                temporal_precision=temporal_precision,
            ),
            encryption_level=encryption_level,
            encrypted=encrypted_flag,
            parent_id=parent_id,
            tags=tags or [],
            content_hash=ContentHasher.hash(content),
        )

        # ── Parent-child wiring
        if parent_id:
            parent = await self._async_storage.get_block(parent_id)
            if parent is not None:
                parent.children_ids.append(block.id)
                await self._async_storage.update_block(
                    parent_id, {"children_ids": parent.children_ids},
                )

        # ── Op-log append (hash-chained)
        op = await self._append_operation(
            action=OpAction.CREATE,
            block_id=block.id,
            data={"content": content, "type": type.value},
        )
        block.op_hash = op.hash
        block.version = 1

        # ── Persist block
        await self._async_storage.save_block(block)

        # ── Embedding generation (best-effort; embed plaintext, not encrypted)
        if self._embedding_provider is not None:
            await self._embed_block(block.id, content)

        # ── Auto-link edges
        await self._auto_link_block_async(block, tags)

        # ── Hook emission — ON_ADD fires after the block is fully
        # persisted (block + metadata + embedding + auto-link).
        # Failures inside hook callbacks are swallowed by HookManager.
        self._hooks.emit(EventType.ON_ADD, {
            "block_id": block.id,
            "block": block,
            "content": content,
            "type": type.value,
        })

        # ── Auto-extract-on-store: derive additional facts from the
        # content we just stored. Two paths:
        #   - background_extract=True → schedule as asyncio.Task,
        #     return immediately. Caller can await `wait_for_extractions`.
        #   - background_extract=False → await synchronously inline.
        # In both cases the recursion guard `_extracting` prevents
        # the extracted store() calls from re-triggering this path.
        if (
            self._auto_extract_on_store
            and not self._extracting
            and self._extract_provider_name is not None
        ):
            if self._background_extract:
                task = asyncio.create_task(
                    self._auto_extract_after_store(block.id, content),
                )
                self._bg_tasks.add(task)
                # Auto-clean: remove from set when done (prevents
                # unbounded growth on long-lived AsyncMemBlock).
                task.add_done_callback(self._bg_tasks.discard)
            else:
                await self._auto_extract_after_store(block.id, content)

        return block

    async def get(
        self, block_id: str, decrypt: bool = True,
    ) -> Block | None:
        """Retrieve a block by ID."""
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.get, block_id, decrypt,
            )
        await self._ensure_initialized()
        block = await self._async_storage.get_block(block_id)
        if block is None or not decrypt:
            return block
        # Decrypt if needed
        if block.encrypted and self._crypto is not None:
            try:
                block.content = self._crypto.unseal(
                    block.content, block.encryption_level,
                )
                block.encrypted = False
            except Exception:
                pass
        return block

    async def update(
        self, block_id: str, **updates: Any,
    ) -> Block | None:
        """Update a block's fields."""
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.update, block_id, **updates,
            )
        await self._ensure_initialized()
        await self._async_storage.update_block(block_id, updates)
        # Op-log
        await self._append_operation(
            action=OpAction.UPDATE,
            block_id=block_id,
            data={"fields": list(updates.keys())},
        )
        block = await self._async_storage.get_block(block_id)
        # ON_UPDATE hook
        if block is not None:
            self._hooks.emit(EventType.ON_UPDATE, {
                "block_id": block_id,
                "block": block,
                "fields": list(updates.keys()),
            })
        return block

    async def delete(
        self, block_id: str, cascade: bool = False,
    ) -> bool:
        """Soft-delete (mark `deleted=True`) a block.

        Set `cascade=True` to also delete children + edges. In
        native mode we delegate the cascade to storage's FK constraints
        for edges; child blocks are deleted via recursive calls.
        """
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.delete, block_id, cascade,
            )
        await self._ensure_initialized()
        block = await self._async_storage.get_block(block_id)
        if block is None:
            return False

        if cascade:
            for child_id in block.children_ids:
                await self.delete(child_id, cascade=True)

        # Soft-delete: mark deleted but keep the row
        await self._async_storage.update_block(block_id, {"deleted": True})

        # Op-log
        await self._append_operation(
            action=OpAction.DELETE,
            block_id=block_id,
            data={"cascade": cascade},
        )

        # ON_DELETE hook — fires on the block we just soft-deleted.
        # The block payload is the pre-delete snapshot so callbacks
        # can see what was removed.
        self._hooks.emit(EventType.ON_DELETE, {
            "block_id": block_id,
            "block": block,
            "cascade": cascade,
        })
        return True

    # ─── Graph Operations ─────────────────────────────────────────────

    async def link(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | str = EdgeRelation.RELATED_TO,
        weight: float = 1.0,
    ) -> None:
        if not self._native_async:
            await asyncio.to_thread(
                self._mem.link, source_id, target_id, relation, weight,
            )
            return
        await self._ensure_initialized()
        if isinstance(relation, str):
            relation = EdgeRelation(relation)
        edge = Edge(
            id=generate_edge_id(),
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
        )
        await self._async_storage.save_edge(edge)
        await self._append_operation(
            action=OpAction.LINK,
            block_id=source_id,
            data={"target_id": target_id, "relation": relation.value},
        )

    async def unlink(
        self,
        source_id: str,
        target_id: str,
        relation: EdgeRelation | str | None = None,
    ) -> int:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.unlink, source_id, target_id, relation,
            )
        await self._ensure_initialized()
        # No bulk-unlink in async storage adapter; gather edges and
        # delete the matches.
        edges = await self._async_storage.get_edges(
            source_id, direction="outgoing",
        )
        rel_value = (
            relation.value if isinstance(relation, EdgeRelation)
            else relation
        )
        removed = 0
        for edge in edges:
            if edge.target_id != target_id:
                continue
            if rel_value is not None and edge.relation.value != rel_value:
                continue
            await self._async_storage.delete_edge(edge.id)
            removed += 1
        return removed

    async def neighbors(
        self,
        block_id: str,
        relation: EdgeRelation | None = None,
    ) -> list[Block]:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.neighbors, block_id, relation,
            )
        await self._ensure_initialized()
        edges = await self._async_storage.get_edges(
            block_id, direction="both",
        )
        rel_value = relation.value if isinstance(relation, EdgeRelation) else None
        neighbor_ids: set[str] = set()
        for edge in edges:
            if rel_value is not None and edge.relation.value != rel_value:
                continue
            neighbor_ids.add(
                edge.target_id if edge.source_id == block_id
                else edge.source_id,
            )
        if not neighbor_ids:
            return []
        blocks = await asyncio.gather(*[
            self._async_storage.get_block(bid) for bid in neighbor_ids
        ])
        return [b for b in blocks if b is not None and not b.deleted]

    async def traverse(
        self, block_id: str, max_depth: int = 3,
    ) -> list[Block]:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.traverse, block_id, max_depth,
            )
        await self._ensure_initialized()
        # Reuse AsyncContextBuilder's BFS helper
        return await self._async_context._async_traverse(block_id, max_depth)

    # ─── Query ────────────────────────────────────────────────────────

    async def query(
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
    ) -> list[Block]:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.query,
                type=type, tags=tags, text_search=text_search,
                related_to=related_to, min_confidence=min_confidence,
                sort_by=sort_by, limit=limit,
                include_decayed=include_decayed, min_strength=min_strength,
                semantic=semantic,
                session_id=session_id, org_id=org_id,
                project_id=project_id, agent_id=agent_id,
                metadata_filters=metadata_filters,
            )
        await self._ensure_initialized()
        results = await self._async_query.query(
            type=type, tags=tags, text_search=text_search,
            related_to=related_to, min_confidence=min_confidence,
            sort_by=sort_by, limit=limit,
            include_decayed=include_decayed, min_strength=min_strength,
            semantic=semantic,
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata_filters=metadata_filters,
        )
        # ON_QUERY hook — fires after retrieval. Useful for audit /
        # query-pattern analytics. Payload mirrors sync MemBlock.
        self._hooks.emit(EventType.ON_QUERY, {
            "query_text": text_search,
            "type": type.value if type is not None else None,
            "result_count": len(results),
            "result_ids": [b.id for b in results],
        })
        return results

    # ─── Context Builder ──────────────────────────────────────────────

    async def build_context(
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
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.build_context,
                query=query, token_budget=token_budget, strategy=strategy,
                include_metadata=include_metadata,
                session_id=session_id, org_id=org_id,
                project_id=project_id, agent_id=agent_id,
                metadata_filters=metadata_filters,
            )
        await self._ensure_initialized()
        return await self._async_context.build_context(
            query=query, token_budget=token_budget, strategy=strategy,
            include_metadata=include_metadata,
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata_filters=metadata_filters,
        )

    # ─── Auto-link / Auto-extract / Messages ─────────────────────────

    async def enable_auto_link(
        self,
        enabled: bool = True,
        max_neighbors: int = 5,
    ) -> None:
        """Toggle automatic graph linking on store(). In legacy mode
        delegates to sync MemBlock; in native mode flips internal
        state used by `store()`."""
        if not self._native_async:
            await asyncio.to_thread(
                self._mem.enable_auto_link, enabled, max_neighbors,
            )
            return
        self._auto_link_enabled = enabled
        self._auto_link_max_neighbors = max_neighbors
        if enabled:
            self._last_stored_id = None
            self._tag_index = {}

    async def add_message(
        self, role: str, content: str,
    ) -> Any:
        """Buffer a chat message; fires extraction every Nth call
        when auto_extract is configured."""
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.add_message, role, content,
            )
        await self._ensure_initialized()
        self._message_buffer.append({"role": role, "content": content})
        self._message_count += 1
        if (
            self._auto_extract
            and self._message_count % self._extract_every == 0
            and len(self._message_buffer) > 0
        ):
            return await self._run_extraction_async()
        return None

    async def flush_extraction(self) -> Any:
        """Force-flush buffered messages through the extractor."""
        if not self._native_async:
            return await asyncio.to_thread(self._mem.flush_extraction)
        await self._ensure_initialized()
        if not self._message_buffer:
            from memblock.extraction import ExtractionResult
            return ExtractionResult(
                provider="none", errors=["No messages in buffer"],
            )
        return await self._run_extraction_async()

    @property
    def extraction_pending(self) -> int:
        """Count of background extraction tasks still running.

        Native mode: tracks `auto_extract_on_store + background_extract`
        tasks via `_bg_tasks`. Returns the number not yet done.
        Legacy mode: forwards to sync MemBlock's worker queue depth.
        """
        if not self._native_async and self._mem is not None:
            return self._mem.extraction_pending
        # `_bg_tasks` auto-removes done tasks via callback, but on a
        # tight read window the callback may not have fired yet —
        # filter for safety.
        return sum(1 for t in self._bg_tasks if not t.done())

    @property
    def has_embeddings(self) -> bool:
        """Whether semantic search is wired up."""
        if not self._native_async and self._mem is not None:
            return self._mem.has_embeddings
        return self._embedding_provider is not None

    # ─── Decay maintenance ───────────────────────────────────────────
    #
    # `DecayEngine.calculate_strength` is pure Python — we reuse it
    # directly in native mode. The maintenance methods (apply_decay,
    # prune, strongest, weakest) need their own async implementations
    # using `await self._async_storage.<method>(...)`.

    async def prune(self, min_strength: float = 0.1) -> list[Block]:
        """Soft-delete blocks whose strength has fallen below the
        threshold (or whose TTL has expired)."""
        if not self._native_async:
            return await asyncio.to_thread(self._mem.prune, min_strength)
        await self._ensure_initialized()
        from memblock.decay import DecayEngine
        decay = DecayEngine(storage=None)  # type: ignore[arg-type]

        blocks = await self._async_storage.get_all_blocks()
        current_time = now_utc()
        pruned: list[Block] = []
        for block in blocks:
            should_prune = False
            if block.metadata.ttl is not None:
                age = (current_time - block.metadata.created_at).total_seconds()
                if age > block.metadata.ttl:
                    should_prune = True
            if not should_prune:
                strength = decay.calculate_strength(block, at_time=current_time)
                if strength < min_strength:
                    should_prune = True
            if should_prune:
                await self._async_storage.update_block(
                    block.id, {"deleted": True},
                )
                pruned.append(block)
        return pruned

    async def strongest(
        self, limit: int = 10,
    ) -> list[tuple[Block, float]]:
        """Top-N memories by decay-adjusted strength, descending."""
        if not self._native_async:
            return await asyncio.to_thread(self._mem.strongest, limit)
        return await self._scored_blocks_by_strength(reverse=True, limit=limit)

    async def weakest(
        self, limit: int = 10,
    ) -> list[tuple[Block, float]]:
        """Bottom-N memories by strength (candidates for pruning)."""
        if not self._native_async:
            return await asyncio.to_thread(self._mem.weakest, limit)
        return await self._scored_blocks_by_strength(reverse=False, limit=limit)

    async def _scored_blocks_by_strength(
        self, reverse: bool, limit: int,
    ) -> list[tuple[Block, float]]:
        """Internal helper — load all blocks, compute strength,
        sort, slice. Used by strongest()/weakest()."""
        await self._ensure_initialized()
        from memblock.decay import DecayEngine
        decay = DecayEngine(storage=None)  # type: ignore[arg-type]
        blocks = await self._async_storage.get_all_blocks()
        scored = [(b, decay.calculate_strength(b)) for b in blocks]
        scored.sort(key=lambda x: x[1], reverse=reverse)
        return scored[:limit]

    # ─── Tamper-evident op-log verify ────────────────────────────────

    async def verify(self) -> TamperReport:
        """Walk the operation log and verify every hash + chain link.

        Mirrors `OpLog.verify` from the sync path but runs the storage
        fetch via async. Hashing logic is identical (SHA-256 over
        `prev_hash | action | data | timestamp`)."""
        if not self._native_async:
            return await asyncio.to_thread(self._mem.verify)
        await self._ensure_initialized()
        ops = await self._async_storage.get_operations()
        if not ops:
            return TamperReport(
                valid=True, total_ops=0,
                message="No operations to verify",
            )
        prev_hash = ""
        for op in ops:
            if op.prev_hash != prev_hash:
                return TamperReport(
                    valid=False, total_ops=len(ops),
                    first_tampered_op=op.id,
                    message=(
                        f"Chain broken at op {op.id}: expected "
                        f"prev_hash={prev_hash!r}, got {op.prev_hash!r}"
                    ),
                )
            payload = (
                f"{op.prev_hash}|{op.action.value}|"
                f"{json.dumps(op.data, sort_keys=True)}|"
                f"{op.timestamp.isoformat()}"
            )
            expected_hash = (
                f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"
            )
            if op.hash != expected_hash:
                return TamperReport(
                    valid=False, total_ops=len(ops),
                    first_tampered_op=op.id,
                    message=(
                        f"Hash mismatch at op {op.id}: expected "
                        f"{expected_hash}, got {op.hash}"
                    ),
                )
            prev_hash = op.hash
        return TamperReport(
            valid=True, total_ops=len(ops),
            message=f"All {len(ops)} operations verified successfully",
        )

    # ─── Sessions ────────────────────────────────────────────────────

    async def get_sessions(self) -> list[str]:
        """All distinct session_ids the user has blocks in."""
        if not self._native_async:
            return await asyncio.to_thread(self._mem.get_sessions)
        await self._ensure_initialized()
        blocks = await self._async_storage.get_all_blocks()
        sessions: set[str] = set()
        for b in blocks:
            if b.metadata.session_id is not None:
                sessions.add(b.metadata.session_id)
        return sorted(sessions)

    async def get_session_history(
        self, session_id: str, limit: int = 100,
    ) -> list[Block]:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.get_session_history, session_id, limit,
            )
        await self._ensure_initialized()
        blocks = await self._async_storage.query_blocks({
            "session_id": session_id, "limit": limit,
            "sort_by": "created_at",
        })
        # Chronological (oldest first)
        blocks.sort(key=lambda b: b.metadata.created_at.isoformat())
        return blocks

    # ─── Standalone Extract / Multi-Hop ──────────────────────────────

    async def extract(
        self, conversation: str, **kwargs: Any,
    ) -> Any:
        """Extract memory blocks from a single conversation string.
        Native mode: same logic as `_run_extraction_async` but takes
        an explicit conversation instead of the buffered messages.
        Persists extracted blocks via native `store()`.
        """
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.extract, conversation, **kwargs,
            )
        await self._ensure_initialized()
        from memblock.extraction import (
            LLMExtractor, ExtractionResult, EXTRACTION_USER_TEMPLATE,
        )

        if self._extract_provider_name is None:
            return ExtractionResult(
                provider="none",
                errors=["extract_provider not configured"],
            )

        provider = await self._init_extractor_provider()
        extractor = LLMExtractor(
            provider=provider,
            min_confidence=self._extract_min_confidence,
        )
        user_prompt = EXTRACTION_USER_TEMPLATE.format(
            conversation=conversation,
        )
        try:
            raw_response = await asyncio.to_thread(
                provider.complete, extractor.system_prompt, user_prompt,
            )
        except Exception as e:
            return ExtractionResult(
                provider=provider.name, errors=[f"LLM call failed: {e}"],
            )
        try:
            extracted = extractor._parse_response(raw_response)
        except Exception as e:
            return ExtractionResult(
                provider=provider.name, raw_response=raw_response,
                errors=[f"Parse failed: {e}"],
            )

        block_ids: list[str] = []
        for item in extracted or []:
            confidence = float(item.get("confidence", 0.5))
            if confidence < self._extract_min_confidence:
                continue
            try:
                block = await self.store(
                    content=item["content"],
                    type=BlockType(item.get("type", "fact")),
                    confidence=confidence,
                    source=SourceType(item.get("source", "inferred")),
                    tags=item.get("tags") or None,
                )
                if block is not None:
                    block_ids.append(block.id)
            except Exception:
                continue

        return ExtractionResult(
            provider=provider.name, raw_response=raw_response,
            block_ids=block_ids, blocks_created=len(block_ids),
        )

    async def extract_messages(
        self, messages: list[dict[str, str]], **kwargs: Any,
    ) -> Any:
        """Extract from a list of `{role, content}` dicts."""
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.extract_messages, messages, **kwargs,
            )
        # Format then forward to extract()
        lines = [
            f"{(m.get('role') or 'user').title()}: {m.get('content') or ''}"
            for m in messages
        ]
        return await self.extract("\n".join(lines), **kwargs)

    async def multi_hop_query(
        self,
        query: str,
        limit: int = 10,
        session_id: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[Block]:
        """Multi-hop iterative retrieval. Native mode runs the same
        4-pass strategy (entity-extract-from-question → standard
        retrieval → entity-extract-from-results + per-entity query
        → graph walk) but with concurrent fetches per hop.
        """
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.multi_hop_query,
                query, limit, session_id, org_id,
                project_id, agent_id, metadata_filters,
            )
        await self._ensure_initialized()

        from memblock.decay import DecayEngine
        decay = DecayEngine(storage=None)  # type: ignore[arg-type]
        scope = dict(
            session_id=session_id or self._session_id,
            org_id=org_id or self._org_id,
            project_id=project_id or self._project_id,
            agent_id=agent_id or self._agent_id,
            metadata_filters=metadata_filters,
        )

        seen_ids: set[str] = set()
        all_scored: list[tuple[Block, float]] = []

        # Hop 0: extract entities from the question itself
        question_entities = _extract_entities_from_text(query)

        # Hop 1: hybrid retrieval
        hop1 = await self._async_query.query(
            text_search=query, sort_by="relevance",
            limit=limit * 3, **scope,
        )
        for block in hop1:
            if block.id not in seen_ids:
                seen_ids.add(block.id)
                strength = decay.calculate_strength(block)
                all_scored.append((block, strength + 1.0))

        if not hop1 and question_entities:
            # Fallback — use question entities directly
            entity_results = await asyncio.gather(*[
                self._async_query.query(
                    text_search=ent, sort_by="relevance",
                    limit=limit, **scope,
                )
                for ent in question_entities[:3]
            ])
            for blocks in entity_results:
                for block in blocks:
                    if block.id not in seen_ids:
                        seen_ids.add(block.id)
                        strength = decay.calculate_strength(block)
                        all_scored.append((block, strength + 0.8))
            if not all_scored:
                return []

        # Hop 2: extract entities from hop1 results, query each
        block_entities = _extract_entities_from_blocks(hop1)
        all_entities = list(dict.fromkeys(question_entities + block_entities))

        if all_entities:
            entity_results = await asyncio.gather(*[
                self._async_query.query(
                    text_search=ent, sort_by="relevance",
                    limit=limit, **scope,
                )
                for ent in all_entities[:10]
            ])
            for ent, blocks in zip(all_entities[:10], entity_results):
                bonus = 0.8 if ent in question_entities else 0.5
                for block in blocks:
                    if block.id not in seen_ids:
                        seen_ids.add(block.id)
                        strength = decay.calculate_strength(block)
                        all_scored.append((block, strength + bonus))

        # Hop 3: graph walk from all retrieved blocks
        graph_candidates: dict[str, int] = {}
        depth_results = await asyncio.gather(*[
            self._async_query._traverse_with_depth(b.id, max_depth=3)
            for b, _ in all_scored
        ])
        for depth_map in depth_results:
            for nid, depth in depth_map.items():
                if nid not in seen_ids:
                    if (nid not in graph_candidates
                            or depth < graph_candidates[nid]):
                        graph_candidates[nid] = depth

        if graph_candidates:
            fetched = await asyncio.gather(*[
                self._async_storage.get_block(nid)
                for nid in graph_candidates
            ])
            for nid, block in zip(graph_candidates.keys(), fetched):
                if block and not block.deleted:
                    seen_ids.add(nid)
                    strength = decay.calculate_strength(block)
                    proximity_bonus = 0.4 / graph_candidates[nid]
                    all_scored.append((block, strength + proximity_bonus))

        all_scored.sort(key=lambda x: x[1], reverse=True)
        return [block for block, _ in all_scored[:limit]]

    # ─── Analytics (org-level question tracking) ─────────────────────

    async def log_question(
        self,
        question: str,
        user_id: str | None = None,
        org_id: str | None = None,
    ) -> Any:
        """Track a question for org-level analytics.
        Native mode delegates to the async adapter's analytics helpers.
        """
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.log_question, question, user_id, org_id,
            )
        await self._ensure_initialized()
        # Lazy import — analytics use is opt-in. `OrgAnalytics`
        # exposes `normalize_question` as a static method, and
        # `NoiseFilter.is_noise` for pre-filter rejection.
        from memblock.analytics import NoiseFilter, OrgAnalytics
        if NoiseFilter().is_noise(question):
            return None
        normalized = OrgAnalytics.normalize_question(question)
        if not normalized:
            return None
        return await self._async_storage.upsert_question_async(
            org_id=org_id or self._org_id or "default",
            normalized_text=normalized,
            user_id=user_id or self._user_id,
            asked_at=now_utc(),
        )

    async def get_top_questions(
        self,
        org_id: str | None = None,
        limit: int = 20,
        **kwargs: Any,
    ) -> list[Any]:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.get_top_questions, org_id, limit, **kwargs,
            )
        await self._ensure_initialized()
        return await self._async_storage.get_top_questions_async(
            org_id=org_id or self._org_id or "default",
            limit=limit,
            since=kwargs.get("since"),
            until=kwargs.get("until"),
        )

    async def get_trending_questions(
        self,
        org_id: str | None = None,
        window_days: int = 7,
        limit: int = 10,
    ) -> list[Any]:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.get_trending_questions, org_id, window_days, limit,
            )
        await self._ensure_initialized()
        # Native v0: simple "top by frequency in window" via
        # get_top_questions_async with `since` filter.
        from datetime import timedelta
        since = now_utc() - timedelta(days=window_days)
        return await self._async_storage.get_top_questions_async(
            org_id=org_id or self._org_id or "default",
            limit=limit, since=since,
        )

    async def get_question_breakdown(
        self,
        question: str,
        org_id: str | None = None,
        granularity: str = "daily",
    ) -> dict[str, Any]:
        if not self._native_async:
            return await asyncio.to_thread(
                self._mem.get_question_breakdown,
                question, org_id, granularity,
            )
        await self._ensure_initialized()
        # Normalize before lookup — table is keyed on normalized_text
        from memblock.analytics import OrgAnalytics
        normalized = OrgAnalytics.normalize_question(question)
        return await self._async_storage.get_question_breakdown_async(
            org_id=org_id or self._org_id or "default",
            question_text=normalized, granularity=granularity,
        )

    async def question_stats(
        self, org_id: str | None = None,
    ) -> dict[str, Any]:
        if not self._native_async:
            return await asyncio.to_thread(self._mem.question_stats, org_id)
        await self._ensure_initialized()
        return await self._async_storage.question_stats_async(
            org_id=org_id or self._org_id or "default",
        )

    @property
    def extraction_stats(self) -> dict[str, int]:
        """Background extraction worker stats. Native mode doesn't
        run a background extractor in v0 — returns zeros."""
        if not self._native_async and self._mem is not None:
            return self._mem.extraction_stats
        return {"pending": 0, "completed": 0, "failed": 0, "total_submitted": 0}

    async def wait_for_extractions(
        self, timeout: float | None = None,
    ) -> None:
        """Block until pending background extractions complete.

        Native mode: awaits every task in `self._bg_tasks` (added
        by `auto_extract_on_store + background_extract=True`) with
        the given timeout. Returns when all complete or timeout
        elapses (whichever first).

        Legacy mode: forwards to sync MemBlock's thread-pool waiter.
        """
        if not self._native_async and self._mem is not None:
            await asyncio.to_thread(
                self._mem.wait_for_extractions, timeout,
            )
            return

        if not self._bg_tasks:
            return
        # Snapshot — `_bg_tasks` mutates as tasks finish (via the
        # done-callback that calls `discard`).
        pending = list(self._bg_tasks)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pass

    # ─── Hooks ────────────────────────────────────────────────────────

    def on(self, event: str, callback: Any) -> None:
        """Register a callback for a memory lifecycle event.

        Both modes:
          - Sync callbacks fire inline.
          - Async callbacks are scheduled as tasks on the running
            event loop (HookManager handles dispatch).

        Native mode emits `ON_ADD` after `store`, `ON_UPDATE` after
        `update`, `ON_DELETE` after `delete`, and `ON_QUERY` after
        `query`. Same lifecycle as sync MemBlock — no behavioural
        regression when flipping the URL to `+asyncpg`.
        """
        # Always register on the local HookManager so native mode
        # has callbacks to fire.
        if isinstance(callback, type(self.on)):  # method — fine
            pass
        if asyncio.iscoroutinefunction(callback):
            self._hooks.register_async(event, callback)
        else:
            self._hooks.register(event, callback)
        # Mirror to legacy sync MemBlock so nothing is lost when
        # users register on AsyncMemBlock then access `.sync`.
        if not self._native_async and self._mem is not None:
            self._mem.on(event, callback)

    # ─── Stats / Export ───────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        if not self._native_async:
            return await asyncio.to_thread(self._mem.stats)
        await self._ensure_initialized()
        all_blocks = await self._async_storage.get_all_blocks(
            include_deleted=True,
        )
        live = [b for b in all_blocks if not b.deleted]
        return {
            "total_blocks": len(all_blocks),
            "live_blocks": len(live),
            "deleted_blocks": len(all_blocks) - len(live),
            "user_id": self._user_id,
            "adapter": self._async_storage.adapter_type,
        }

    async def export_markdown(self) -> str:
        if not self._native_async:
            return await asyncio.to_thread(self._mem.export_markdown)
        await self._ensure_initialized()
        blocks = await self._async_storage.get_all_blocks()
        lines = ["# Memory Export"]
        for b in blocks:
            lines.append(f"- [{b.type.value}] {b.content}")
        return "\n".join(lines)

    # ─── Lifecycle ────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._native_async:
            await self._async_storage.close()
        else:
            await asyncio.to_thread(self._mem.close)

    async def __aenter__(self) -> AsyncMemBlock:
        if self._native_async:
            await self._ensure_initialized()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        if self._native_async:
            return (
                f"<AsyncMemBlock(native, user={self._user_id!r}, "
                f"storage=postgresql+asyncpg://...)>"
            )
        return f"Async{repr(self._mem)}"

    # ─── Native-mode helpers (private) ────────────────────────────────

    async def _append_operation(
        self,
        action: OpAction,
        block_id: str,
        data: dict[str, Any],
    ) -> Operation:
        """Hash-chained op-log append. Mirrors `OpLog.append` from
        the sync path."""
        # Lazy clock init — read the last op from storage on first
        # call, then track in memory.
        if self._op_clock is None:
            last = await self._async_storage.get_last_operation()
            if last is None:
                self._op_clock = 0
                self._last_op_hash = ""
            else:
                self._op_clock = last.clock + 1
                self._last_op_hash = last.hash

        timestamp = now_utc()
        prev_hash = self._last_op_hash
        payload = (
            f"{prev_hash}|{action.value}|"
            f"{json.dumps(data, sort_keys=True)}|{timestamp.isoformat()}"
        )
        op_hash = f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"

        op = Operation(
            id=generate_op_id(),
            block_id=block_id,
            author=self._author,
            clock=self._op_clock,
            action=action,
            data=data,
            hash=op_hash,
            prev_hash=prev_hash,
            timestamp=timestamp,
        )
        await self._async_storage.save_operation(op)
        self._op_clock += 1
        self._last_op_hash = op_hash
        return op

    async def _embed_block(
        self, block_id: str, content: str,
    ) -> None:
        """Generate + persist an embedding for a block. Best-effort —
        failures are swallowed so they don't break store()."""
        if self._embedding_provider is None:
            return
        try:
            from memblock.embeddings import pack_embedding
            vectors = await asyncio.to_thread(
                self._embedding_provider.embed, [content],
            )
            if vectors:
                await self._async_storage.save_embedding(
                    block_id, pack_embedding(vectors[0]),
                )
        except Exception:
            pass  # Embedding failure should not break store

    async def _auto_link_block_async(
        self, block: Block, tags: list[str] | None,
    ) -> None:
        """Sequential + tag-based auto-link, mirror of
        `MemBlock._auto_link_block`."""
        if not self._auto_link_enabled:
            return

        linked_ids: set[str] = set()
        # 1. Sequential link
        if (self._last_stored_id is not None
                and self._last_stored_id != block.id):
            try:
                await self.link(
                    block.id, self._last_stored_id,
                    relation=EdgeRelation.RELATED_TO, weight=0.8,
                )
                linked_ids.add(self._last_stored_id)
            except Exception:
                pass

        # 2. Tag-based links
        if tags:
            candidates: dict[str, int] = {}
            for tag in tags:
                for prev_id in self._tag_index.get(tag, []):
                    if prev_id != block.id and prev_id not in linked_ids:
                        candidates[prev_id] = candidates.get(prev_id, 0) + 1
            sorted_candidates = sorted(
                candidates.items(), key=lambda x: x[1], reverse=True,
            )
            for prev_id, shared_count in (
                sorted_candidates[: self._auto_link_max_neighbors]
            ):
                weight = min(1.0, 0.3 + shared_count * 0.2)
                try:
                    await self.link(
                        block.id, prev_id,
                        relation=EdgeRelation.RELATED_TO, weight=weight,
                    )
                    linked_ids.add(prev_id)
                except Exception:
                    pass

            for tag in tags:
                self._tag_index.setdefault(tag, []).append(block.id)
                if len(self._tag_index[tag]) > 50:
                    self._tag_index[tag] = self._tag_index[tag][-50:]

        self._last_stored_id = block.id

    async def _init_extractor_provider(self) -> Any:
        """Lazily build the LLM provider for conflict resolution +
        auto-extraction. Mirrors `MemBlock._init_extractor` but runs
        the constructor in a thread (provider init may import slow
        SDKs like `openai` / `anthropic`)."""
        from memblock.extraction import (
            OpenAIProvider, AnthropicProvider, GeminiProvider,
        )
        provider_name = self._extract_provider_name
        api_key = self._extract_api_key
        model = self._extract_model

        def _build():
            if provider_name == "openai":
                if not api_key:
                    raise ValueError(
                        "extract_api_key is required for OpenAI extraction"
                    )
                return OpenAIProvider(
                    api_key=api_key, model=model or "gpt-4o-mini",
                )
            if provider_name == "anthropic":
                if not api_key:
                    raise ValueError(
                        "extract_api_key is required for Anthropic extraction"
                    )
                return AnthropicProvider(
                    api_key=api_key,
                    model=model or "claude-3-5-haiku-latest",
                )
            if provider_name == "gemini":
                if not api_key:
                    raise ValueError(
                        "extract_api_key is required for Gemini extraction"
                    )
                return GeminiProvider(
                    api_key=api_key, model=model or "gemini-2.0-flash",
                )
            raise ValueError(f"Unknown extract_provider: {provider_name}")

        return await asyncio.to_thread(_build)

    async def _auto_extract_after_store(
        self, source_block_id: str, content: str,
    ) -> None:
        """Run extraction on a freshly-stored block's content and
        link any derived blocks back via DERIVED_FROM.

        Used by the `auto_extract_on_store` path inside `store()`.
        Recursion guard via `_extracting` so the extracted blocks'
        own store calls don't re-trigger extraction.

        Failures are swallowed — auto-extraction is best-effort,
        a crash here MUST NOT break the original store() that the
        user called.
        """
        if self._extract_provider_name is None:
            return
        self._extracting = True
        try:
            result = await self.extract(content)
        except Exception:
            return
        finally:
            self._extracting = False

        # Link extracted blocks back to the original via DERIVED_FROM
        # — same convention as sync MemBlock.
        for extracted_id in (getattr(result, "block_ids", None) or []):
            try:
                await self.link(
                    extracted_id, source_block_id,
                    relation=EdgeRelation.DERIVED_FROM,
                )
            except Exception:
                pass

    async def _run_extraction_async(self) -> Any:
        """Async equivalent of `MemBlock._run_extraction()`.

        Strategy:
          1. Build the provider (lazy, cached)
          2. Format buffered messages to a conversation string
          3. Call provider.complete via to_thread (sync HTTP)
          4. Parse the response with the sync extractor's parser
             (`_parse_response` — pure Python, no I/O)
          5. For each parsed item, await `self.store(...)`
          6. Clear the buffer on success; preserve on failure
             (caller retries via `flush_extraction`).

        Recursion guard via `_extracting` so the extractor's writes
        don't re-trigger conflict resolution or further extraction.
        """
        from memblock.extraction import (
            LLMExtractor, ExtractionResult, EXTRACTION_USER_TEMPLATE,
        )

        if not self._message_buffer:
            return ExtractionResult(provider="none")

        self._extracting = True
        try:
            provider = await self._init_extractor_provider()
            extractor = LLMExtractor(
                provider=provider,
                min_confidence=self._extract_min_confidence,
            )

            # ── Format conversation
            lines: list[str] = []
            for msg in self._message_buffer:
                role = (msg.get("role") or "user").title()
                content = msg.get("content") or ""
                lines.append(f"{role}: {content}")
            conversation = "\n".join(lines)

            # ── Call LLM (sync HTTP — wrap)
            user_prompt = EXTRACTION_USER_TEMPLATE.format(
                conversation=conversation,
            )
            try:
                raw_response = await asyncio.to_thread(
                    provider.complete, extractor.system_prompt, user_prompt,
                )
            except Exception as e:
                return ExtractionResult(
                    provider=provider.name,
                    errors=[f"LLM call failed: {e}"],
                )

            # ── Parse (pure Python)
            try:
                extracted = extractor._parse_response(raw_response)
            except Exception as e:
                return ExtractionResult(
                    provider=provider.name,
                    raw_response=raw_response,
                    errors=[f"Parse failed: {e}"],
                )

            # ── Persist via native async store
            block_ids: list[str] = []
            for item in extracted or []:
                confidence = float(item.get("confidence", 0.5))
                if confidence < self._extract_min_confidence:
                    continue
                try:
                    block_type = BlockType(item.get("type", "fact"))
                    source = SourceType(item.get("source", "inferred"))
                    block = await self.store(
                        content=item["content"],
                        type=block_type,
                        confidence=confidence,
                        source=source,
                        tags=item.get("tags") or None,
                    )
                    if block is not None:
                        block_ids.append(block.id)
                except Exception:
                    continue

            # Success — clear the buffer
            self._message_buffer.clear()
            return ExtractionResult(
                provider=provider.name,
                raw_response=raw_response,
                block_ids=block_ids,
                blocks_created=len(block_ids),
            )

        except Exception as e:
            # Don't clear on outer failure; caller can retry.
            return ExtractionResult(
                provider="error",
                errors=[f"extraction failed: {e}"],
            )
        finally:
            self._extracting = False
