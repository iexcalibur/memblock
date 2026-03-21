# MemBlock SDK — Typed block tree + knowledge graph memory for AI agents
"""
MemBlock: Structured memory SDK for AI agents.

Usage:
    from memblock import MemBlock, BlockType, SourceType

    mem = MemBlock(storage="sqlite:///./memory.db")
    block = mem.store(content="User prefers Python", type=BlockType.PREFERENCE)
    mem.link(block.id, other_block.id, relation="supports")
    results = mem.query(type=BlockType.PREFERENCE)
    context = mem.build_context(query="what does the user prefer?", token_budget=4000)
    mem.verify()  # check tamper detection
"""

__version__ = "0.8.2"

from memblock.block import Block
from memblock.context import ContextBuilder
from memblock.crypto import CryptoLayer, CryptoLayerWithPassphrase
from memblock.decay import DecayEngine
from memblock.dedup import DuplicatePolicy
from memblock.analytics import OrgAnalytics, QuestionRecord, NoiseFilter
from memblock.errors import (
    AnalyticsError,
    BlockNotFoundError,
    ConflictResolutionError,
    DuplicateBlockError,
    EncryptionError,
    ExtractionError,
    LicenseError,
    MemBlockError,
    MigrationError,
    StorageError,
    ValidationError,
)
from memblock.graph import GraphIndex
from memblock.hooks import EventType, HookManager
from memblock.ops import OpLog, TamperReport
from memblock.query import QueryEngine
from memblock.schema import BlockSchema, SchemaValidationError
from memblock.storage.base import StorageAdapter
from memblock.storage.sqlite import SQLiteAdapter
from memblock.store import BlockStore
from memblock.types import (
    BlockMetadata,
    BlockType,
    Edge,
    EdgeRelation,
    EncryptionLevel,
    OpAction,
    Operation,
    SourceType,
)

from memblock.context import ContextConfidence
from memblock.memblock import MemBlock
from memblock.multihop import MultiHopRetriever, MultiHopResult
from memblock.temporal import TemporalQueryParser, TemporalConstraint
from memblock.async_memblock import AsyncMemBlock
from memblock.background import BackgroundExtractor
from memblock.pool import AsyncMemBlockPool, MemBlockPool

__all__ = [
    "AnalyticsError",
    "AsyncMemBlock",
    "AsyncMemBlockPool",
    "BackgroundExtractor",
    "MemBlock",
    "MemBlockPool",
    "NoiseFilter",
    "OrgAnalytics",
    "QuestionRecord",
    "Block",
    "BlockMetadata",
    "BlockNotFoundError",
    "BlockSchema",
    "BlockStore",
    "BlockType",
    "ConflictResolutionError",
    "ContextBuilder",
    "ContextConfidence",
    "CryptoLayer",
    "CryptoLayerWithPassphrase",
    "DecayEngine",
    "DuplicateBlockError",
    "DuplicatePolicy",
    "Edge",
    "EdgeRelation",
    "EncryptionError",
    "EncryptionLevel",
    "EventType",
    "ExtractionError",
    "GraphIndex",
    "HookManager",
    "LicenseError",
    "MemBlockError",
    "MigrationError",
    "MultiHopResult",
    "MultiHopRetriever",
    "OpAction",
    "OpLog",
    "Operation",
    "QueryEngine",
    "SchemaValidationError",
    "SourceType",
    "SQLiteAdapter",
    "StorageAdapter",
    "StorageError",
    "TamperReport",
    "TemporalConstraint",
    "TemporalQueryParser",
    "ValidationError",
]
