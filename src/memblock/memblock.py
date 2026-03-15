"""MemBlock — the main facade class composing all SDK components."""

from __future__ import annotations

from typing import Any

from memblock.block import Block
from memblock.context import ContextBuilder
from memblock.crypto import CryptoLayerWithPassphrase
from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
from memblock.ops import OpLog, TamperReport
from memblock.query import QueryEngine
from memblock.schema import SchemaValidationError
from memblock.storage.base import StorageAdapter
from memblock.storage.sqlite import SQLiteAdapter
from memblock.store import BlockStore
from memblock.types import (
    BlockType,
    EdgeRelation,
    EncryptionLevel,
    SourceType,
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
        embeddings: bool | str = False,
        embeddings_api_key: str | None = None,
        embeddings_model: str | None = None,
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
                - "gemini": Gemini text-embedding-004 (requires embeddings_api_key)
            embeddings_api_key: API key for OpenAI/Gemini embedding providers.
            embeddings_model: Override the default embedding model name.
        """
        self._user_id = user_id

        # Parse storage URI
        self._storage = self._create_storage(storage, user_id)
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
        self._query = QueryEngine(
            self._storage, self._graph, self._decay,
            embedding_provider=self._embedding_provider,
        )
        self._context = ContextBuilder(self._storage, self._query, self._graph, self._decay)

    @staticmethod
    def _create_storage(uri: str, user_id: str = "default") -> StorageAdapter:
        """Create a storage adapter from a URI string."""
        if uri.startswith("postgresql://") or uri.startswith("postgres://"):
            try:
                from memblock.storage.postgresql import PostgreSQLAdapter
            except ImportError:
                raise ImportError(
                    "PostgreSQL adapter requires psycopg. "
                    "Install with: pip install memblock[postgres]"
                )
            return PostgreSQLAdapter(dsn=uri, user_id=user_id)
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
                    model=model or "text-embedding-004",
                )
            else:
                raise ValueError(
                    f"Unknown embeddings provider: {embeddings}. "
                    "Use True (local), 'openai', or 'gemini'."
                )

        return None

    # ─── Store Operations ─────────────────────────────────────────────────

    def store(
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
    ) -> Block:
        """
        Store a new memory block.

        Args:
            content: The memory content text
            type: Block type (FACT, PREFERENCE, EVENT, ENTITY, RELATION)
            confidence: Confidence score (0.0-1.0)
            source: How the memory was acquired
            tags: Categorization tags
            parent_id: Parent block ID for tree structure
            encryption_level: NONE, STANDARD, or SENSITIVE
            decay_rate: How fast this memory fades (0 = never)
            ttl: Time-to-live in seconds (None = permanent)

        Returns:
            The created Block.
        """
        # Encrypt if needed
        stored_content = content
        encrypted = False
        if encryption_level != EncryptionLevel.NONE and self._crypto.enabled:
            stored_content = self._crypto.seal(content, encryption_level)
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
        )

        if encrypted:
            self._storage.update_block(block.id, {"encrypted": True})
            block.encrypted = True

        # Generate and store embedding (if provider available)
        # Use original plaintext content for embedding, not encrypted content
        if self._embedding_provider is not None:
            self._embed_block(block.id, content)

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
            block.content = self._crypto.open(block.content, block.encryption_level)

        return block

    def update(self, block_id: str, **updates: Any) -> Block | None:
        """Update a block's fields."""
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

        return result

    def delete(self, block_id: str, cascade: bool = False) -> bool:
        """Soft-delete a block."""
        result = self._store.delete(block_id, cascade=cascade)
        # Remove embedding on delete
        if result:
            self._storage.delete_embedding(block_id)
        return result

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
        semantic: bool = True,
    ) -> list[Block]:
        """
        Query memory blocks with structured filters.

        Args:
            type: Filter by block type
            tags: Filter by tags (match any)
            text_search: Full-text search (FTS + optional vector hybrid)
            related_to: Block ID — find graph-connected blocks
            min_confidence: Minimum confidence threshold
            sort_by: 'relevance', 'recency', 'access_count', 'strength'
            limit: Maximum results
            semantic: Enable hybrid search when embeddings are available (default True)

        Returns:
            List of matching blocks.
        """
        return self._query.query(
            type=type,
            tags=tags,
            text_search=text_search,
            related_to=related_to,
            min_confidence=min_confidence,
            sort_by=sort_by,
            limit=limit,
            semantic=semantic,
        )

    # ─── Context Builder ──────────────────────────────────────────────────

    def build_context(
        self,
        query: str | None = None,
        token_budget: int = 4000,
        strategy: str = "relevance",
        include_metadata: bool = True,
    ) -> str:
        """
        Build LLM-ready context from relevant memory blocks.

        Args:
            query: What to search for
            token_budget: Maximum tokens in output
            strategy: 'relevance', 'graph_walk', or 'type_grouped'
            include_metadata: Include confidence/source in output

        Returns:
            Formatted string for LLM context injection.
        """
        return self._context.build_context(
            query=query,
            token_budget=token_budget,
            strategy=strategy,
            include_metadata=include_metadata,
        )

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
            raise ValueError("api_key is required for auto-extraction")

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
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

        extractor = LLMExtractor(provider=llm_provider)
        return extractor.extract(conversation, memblock=self)

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
            raise ValueError("api_key is required for auto-extraction")

        if provider == "openai":
            llm_provider = OpenAIProvider(api_key=api_key, model=model or "gpt-4o-mini")
        elif provider == "anthropic":
            llm_provider = AnthropicProvider(api_key=api_key, model=model or "claude-sonnet-4-20250514")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        extractor = LLMExtractor(provider=llm_provider)
        return extractor.extract_from_messages(messages, memblock=self)

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

    # ─── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get statistics about the memory store."""
        blocks = self._storage.get_all_blocks()
        all_blocks = self._storage.get_all_blocks(include_deleted=True)

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
        """Close the storage connection."""
        self._storage.close()

    def __enter__(self) -> MemBlock:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        stats = self.stats()
        emb = " +embeddings" if self.has_embeddings else ""
        return f"MemBlock(blocks={stats['total_blocks']}, edges={stats['total_edges']}{emb})"
