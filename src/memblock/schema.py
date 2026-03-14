"""Schema validation for MemBlock blocks and edges."""

from __future__ import annotations

from memblock.block import Block
from memblock.types import BlockType, EdgeRelation, EncryptionLevel, SourceType


class SchemaValidationError(Exception):
    """Raised when a block or edge fails validation."""
    pass


class BlockSchema:
    """
    Validates blocks and edges against MemBlock rules.

    Validation rules:
    - confidence must be between 0.0 and 1.0
    - content must not be empty
    - block type must be valid
    - decay_rate must be >= 0
    - edge relations follow type constraints
    """

    # Which block types can be parents of which
    VALID_PARENTS: dict[BlockType, list[BlockType | None]] = {
        BlockType.FACT: [None, BlockType.ENTITY, BlockType.FACT],
        BlockType.PREFERENCE: [None, BlockType.ENTITY, BlockType.PREFERENCE],
        BlockType.EVENT: [None, BlockType.ENTITY, BlockType.EVENT],
        BlockType.ENTITY: [None, BlockType.ENTITY],
        BlockType.RELATION: [None, BlockType.ENTITY],
    }

    # Edge relation constraints: which relations are valid between which types
    # None means any type is allowed
    EDGE_CONSTRAINTS: dict[EdgeRelation, tuple[list[BlockType] | None, list[BlockType] | None]] = {
        EdgeRelation.CONTRADICTS: (None, None),  # any type can contradict any type
        EdgeRelation.SUPPORTS: (None, None),
        EdgeRelation.ABOUT: (None, [BlockType.ENTITY]),  # 'about' must target an entity
        EdgeRelation.CAUSED_BY: ([BlockType.EVENT], [BlockType.EVENT, BlockType.FACT]),
        EdgeRelation.RELATED_TO: (None, None),
        EdgeRelation.PART_OF: (None, [BlockType.ENTITY]),
        EdgeRelation.DERIVED_FROM: (None, None),
        EdgeRelation.SUPERSEDES: (None, None),
    }

    @classmethod
    def validate_block(cls, block: Block) -> None:
        """
        Validate a block against schema rules.

        Raises SchemaValidationError if validation fails.
        """
        # Content must not be empty
        if not block.content or not block.content.strip():
            raise SchemaValidationError("Block content must not be empty")

        # Block type must be valid
        if not isinstance(block.type, BlockType):
            raise SchemaValidationError(f"Invalid block type: {block.type}")

        # Confidence must be 0.0 - 1.0
        if not 0.0 <= block.metadata.confidence <= 1.0:
            raise SchemaValidationError(
                f"Confidence must be between 0.0 and 1.0, got {block.metadata.confidence}"
            )

        # Source must be valid
        if not isinstance(block.metadata.source, SourceType):
            raise SchemaValidationError(f"Invalid source type: {block.metadata.source}")

        # Decay rate must be non-negative
        if block.metadata.decay_rate < 0:
            raise SchemaValidationError(
                f"Decay rate must be >= 0, got {block.metadata.decay_rate}"
            )

        # TTL must be positive if set
        if block.metadata.ttl is not None and block.metadata.ttl <= 0:
            raise SchemaValidationError(
                f"TTL must be positive if set, got {block.metadata.ttl}"
            )

        # Encryption level must be valid
        if not isinstance(block.encryption_level, EncryptionLevel):
            raise SchemaValidationError(f"Invalid encryption level: {block.encryption_level}")

        # Tags must be strings
        for tag in block.tags:
            if not isinstance(tag, str) or not tag.strip():
                raise SchemaValidationError(f"Tags must be non-empty strings, got {tag!r}")

    @classmethod
    def validate_parent_child(
        cls,
        child_type: BlockType,
        parent_type: BlockType | None,
    ) -> None:
        """
        Validate that a parent-child relationship is allowed.

        Raises SchemaValidationError if the relationship is invalid.
        """
        valid_parents = cls.VALID_PARENTS.get(child_type, [None])
        if parent_type not in valid_parents:
            raise SchemaValidationError(
                f"Block type {child_type.value} cannot have parent of type "
                f"{parent_type.value if parent_type else 'None'}. "
                f"Valid parents: {[p.value if p else 'root' for p in valid_parents]}"
            )

    @classmethod
    def validate_edge(
        cls,
        relation: EdgeRelation,
        source_type: BlockType,
        target_type: BlockType,
    ) -> None:
        """
        Validate that an edge relation is allowed between the given block types.

        Raises SchemaValidationError if the edge is invalid.
        """
        constraint = cls.EDGE_CONSTRAINTS.get(relation)
        if constraint is None:
            return  # no constraints for unknown relations

        allowed_sources, allowed_targets = constraint

        if allowed_sources is not None and source_type not in allowed_sources:
            raise SchemaValidationError(
                f"Edge relation {relation.value} not allowed from block type {source_type.value}. "
                f"Allowed source types: {[t.value for t in allowed_sources]}"
            )

        if allowed_targets is not None and target_type not in allowed_targets:
            raise SchemaValidationError(
                f"Edge relation {relation.value} not allowed to block type {target_type.value}. "
                f"Allowed target types: {[t.value for t in allowed_targets]}"
            )
