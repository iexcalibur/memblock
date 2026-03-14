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

__version__ = "0.1.0"

from memblock.block import Block
from memblock.context import ContextBuilder
from memblock.crypto import CryptoLayer, CryptoLayerWithPassphrase
from memblock.decay import DecayEngine
from memblock.graph import GraphIndex
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

from memblock.memblock import MemBlock

__all__ = [
    "MemBlock",
    "Block",
    "BlockMetadata",
    "BlockSchema",
    "BlockStore",
    "BlockType",
    "ContextBuilder",
    "CryptoLayer",
    "CryptoLayerWithPassphrase",
    "DecayEngine",
    "Edge",
    "EdgeRelation",
    "EncryptionLevel",
    "GraphIndex",
    "OpAction",
    "OpLog",
    "Operation",
    "QueryEngine",
    "SchemaValidationError",
    "SourceType",
    "SQLiteAdapter",
    "StorageAdapter",
    "TamperReport",
]
