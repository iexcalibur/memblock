"""BlockStore — the main CRUD layer for MemBlock."""

from __future__ import annotations

from typing import Any

from memblock.block import Block
from memblock.dedup import ContentHasher
from memblock.errors import BlockNotFoundError, StorageError
from memblock.ops import OpLog
from memblock.schema import BlockSchema, SchemaValidationError
from memblock.storage.base import StorageAdapter
from memblock.types import (
    BlockMetadata,
    BlockType,
    EncryptionLevel,
    OpAction,
    SourceType,
    generate_block_id,
    now_utc,
)


class BlockStore:
    """
    Main CRUD interface for memory blocks.

    Every mutation is validated, logged to OpLog, and persisted via StorageAdapter.
    """

    def __init__(
        self,
        storage: StorageAdapter,
        author: str = "agent",
    ) -> None:
        self.storage = storage
        self.op_log = OpLog(storage, author=author)

    def create(
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
        created_by: str | None = None,
        session_id: str | None = None,
    ) -> Block:
        """
        Create a new memory block.

        Validates the block, logs the operation, and persists it.
        """
        block = Block(
            id=generate_block_id(),
            type=type,
            content=content,
            metadata=BlockMetadata(
                confidence=confidence,
                source=source,
                created_at=now_utc(),
                created_by=created_by or self.op_log.author,
                decay_rate=decay_rate,
                ttl=ttl,
                session_id=session_id,
            ),
            encryption_level=encryption_level,
            parent_id=parent_id,
            tags=tags or [],
            content_hash=ContentHasher.hash(content),
        )

        # Validate
        BlockSchema.validate_block(block)

        # Validate parent-child if parent exists
        if parent_id:
            parent = self.storage.get_block(parent_id)
            if parent is None:
                raise SchemaValidationError(f"Parent block {parent_id} not found")
            BlockSchema.validate_parent_child(block.type, parent.type)

            # Add this block to parent's children
            parent.children_ids.append(block.id)
            self.storage.update_block(parent_id, {"children_ids": parent.children_ids})

        # Log operation
        op = self.op_log.append(
            action=OpAction.CREATE,
            block_id=block.id,
            data={"content": content, "type": type.value},
        )
        block.op_hash = op.hash
        block.version = 1

        # Persist
        try:
            self.storage.save_block(block)
        except (StorageError, SchemaValidationError):
            raise
        except Exception as e:
            raise StorageError(f"Failed to save block {block.id}: {e}") from e
        return block

    def get(self, block_id: str) -> Block | None:
        """
        Retrieve a block by ID.

        Increments access_count and updates last_accessed.
        """
        try:
            block = self.storage.get_block(block_id)
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to retrieve block {block_id}: {e}") from e
        if block is None or block.deleted:
            return None

        # Touch — update access stats
        block.metadata.access_count += 1
        block.metadata.last_accessed = now_utc()
        try:
            self.storage.update_block(block_id, {
                "access_count": block.metadata.access_count,
                "last_accessed": block.metadata.last_accessed,
            })
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update access stats for block {block_id}: {e}") from e

        return block

    def get_without_touch(self, block_id: str) -> Block | None:
        """Retrieve a block without updating access stats."""
        block = self.storage.get_block(block_id)
        if block is None or block.deleted:
            return None
        return block

    def get_or_raise(self, block_id: str) -> Block:
        """Retrieve a block by ID, raising BlockNotFoundError if not found."""
        block = self.get(block_id)
        if block is None:
            raise BlockNotFoundError(f"Block {block_id} not found")
        return block

    def update(self, block_id: str, **updates: Any) -> Block | None:
        """
        Update a block's fields.

        Validates, logs operation, increments version.
        """
        block = self.storage.get_block(block_id)
        if block is None:
            return None

        # Apply updates to a copy for validation
        for key, value in updates.items():
            if key == "content":
                block.content = value
            elif key == "type":
                block.type = value if isinstance(value, BlockType) else BlockType(value)
            elif key == "tags":
                block.tags = value
            elif key == "confidence":
                block.metadata.confidence = value
            elif key == "source":
                block.metadata.source = value if isinstance(value, SourceType) else SourceType(value)
            elif key == "decay_rate":
                block.metadata.decay_rate = value
            elif key == "ttl":
                block.metadata.ttl = value

        # Validate
        BlockSchema.validate_block(block)

        # Log operation
        op = self.op_log.append(
            action=OpAction.UPDATE,
            block_id=block_id,
            data=updates,
        )

        # Update version and hash
        block.version += 1
        block.op_hash = op.hash

        # Persist all changes
        update_dict = dict(updates)
        update_dict["version"] = block.version
        update_dict["op_hash"] = block.op_hash
        try:
            self.storage.update_block(block_id, update_dict)
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update block {block_id}: {e}") from e

        return block

    def delete(self, block_id: str, cascade: bool = False) -> bool:
        """
        Delete a block (soft delete by default).

        cascade=True: also deletes all children recursively.
        cascade=False: reparents children to deleted block's parent.
        """
        try:
            block = self.storage.get_block(block_id)
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to retrieve block {block_id} for deletion: {e}") from e
        if block is None:
            return False

        if cascade:
            # Delete children recursively
            for child_id in block.children_ids:
                self.delete(child_id, cascade=True)
        else:
            # Reparent children to this block's parent
            for child_id in block.children_ids:
                self.storage.update_block(child_id, {"parent_id": block.parent_id})

        # Remove from parent's children list
        if block.parent_id:
            parent = self.storage.get_block(block.parent_id)
            if parent and block_id in parent.children_ids:
                parent.children_ids.remove(block_id)
                self.storage.update_block(parent.id, {"children_ids": parent.children_ids})

        # Log operation
        self.op_log.append(
            action=OpAction.DELETE,
            block_id=block_id,
            data={"cascade": cascade},
        )

        # Clean up edges
        self.storage.delete_edges_for_block(block_id)

        # Soft delete
        self.storage.update_block(block_id, {"deleted": True})
        return True

    def hard_delete(self, block_id: str) -> bool:
        """Permanently remove a block from storage."""
        try:
            block = self.storage.get_block(block_id)
            if block is None:
                return False

            self.storage.delete_edges_for_block(block_id)
            self.storage.delete_block(block_id)
        except StorageError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to hard-delete block {block_id}: {e}") from e
        return True

    # ─── Tree Operations ──────────────────────────────────────────────────

    def get_children(self, block_id: str) -> list[Block]:
        """Get all children of a block."""
        block = self.storage.get_block(block_id)
        if block is None:
            return []
        children = []
        for child_id in block.children_ids:
            child = self.storage.get_block(child_id)
            if child and not child.deleted:
                children.append(child)
        return children

    def get_parent(self, block_id: str) -> Block | None:
        """Get the parent of a block."""
        block = self.storage.get_block(block_id)
        if block is None or block.parent_id is None:
            return None
        return self.storage.get_block(block.parent_id)

    def move(self, block_id: str, new_parent_id: str | None) -> bool:
        """Move a block to a new parent."""
        block = self.storage.get_block(block_id)
        if block is None:
            return False

        # Validate new parent
        if new_parent_id:
            new_parent = self.storage.get_block(new_parent_id)
            if new_parent is None:
                raise SchemaValidationError(f"New parent {new_parent_id} not found")
            BlockSchema.validate_parent_child(block.type, new_parent.type)

        # Remove from old parent
        if block.parent_id:
            old_parent = self.storage.get_block(block.parent_id)
            if old_parent and block_id in old_parent.children_ids:
                old_parent.children_ids.remove(block_id)
                self.storage.update_block(old_parent.id, {"children_ids": old_parent.children_ids})

        # Add to new parent
        if new_parent_id:
            new_parent = self.storage.get_block(new_parent_id)
            if new_parent:
                new_parent.children_ids.append(block_id)
                self.storage.update_block(new_parent_id, {"children_ids": new_parent.children_ids})

        self.storage.update_block(block_id, {"parent_id": new_parent_id})
        return True

    # ─── Bulk Queries ─────────────────────────────────────────────────────

    def get_by_type(self, block_type: BlockType) -> list[Block]:
        """Get all blocks of a specific type."""
        return self.storage.query_blocks({"type": block_type})

    def get_by_tags(self, tags: list[str]) -> list[Block]:
        """Get all blocks matching any of the given tags."""
        return self.storage.query_blocks({"tags": tags})
