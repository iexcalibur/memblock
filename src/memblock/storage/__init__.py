from memblock.storage.base import StorageAdapter
from memblock.storage.sqlite import SQLiteAdapter

# PostgreSQL adapter is optional — only import if psycopg is available
try:
    from memblock.storage.postgresql import PostgreSQLAdapter
    __all__ = ["StorageAdapter", "SQLiteAdapter", "PostgreSQLAdapter"]
except ImportError:
    __all__ = ["StorageAdapter", "SQLiteAdapter"]
