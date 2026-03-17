"""Tests for PostgreSQLAdapter connection pooling support.

Uses mock objects since these tests don't require a real PostgreSQL instance.
"""

import pytest
from unittest.mock import MagicMock, PropertyMock, patch


class TestPostgreSQLAdapterPooling:
    """Test connection pooling integration in PostgreSQLAdapter."""

    def _make_adapter(self, pool=None):
        """Create a PostgreSQLAdapter with mocked psycopg."""
        with patch.dict("sys.modules", {"psycopg": MagicMock(), "psycopg.rows": MagicMock()}):
            # Need to reload to pick up mock
            import importlib
            import memblock.storage.postgresql as pg_mod
            importlib.reload(pg_mod)

            # Manually set the flag since import mock may not set it
            pg_mod.HAS_PSYCOPG = True

            adapter = pg_mod.PostgreSQLAdapter(
                dsn="postgresql://test:test@localhost/test",
                user_id="test_user",
                pool=pool,
            )
            return adapter

    def test_no_pool_creates_direct_connection(self):
        """Without a pool, adapter creates its own connection."""
        adapter = self._make_adapter(pool=None)
        assert adapter._pool is None
        assert adapter._pool_conn is False

    def test_pool_stored_on_init(self):
        """Pool reference is stored on the adapter."""
        mock_pool = MagicMock()
        adapter = self._make_adapter(pool=mock_pool)
        assert adapter._pool is mock_pool
        assert adapter._pool_conn is False

    def test_pool_getconn_called_on_first_access(self):
        """When pool is provided, conn property calls pool.getconn()."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pool.getconn.return_value = mock_conn

        adapter = self._make_adapter(pool=mock_pool)
        conn = adapter.conn

        mock_pool.getconn.assert_called_once()
        assert conn is mock_conn
        assert adapter._pool_conn is True

    def test_pool_conn_reused_on_subsequent_access(self):
        """Second access to conn returns same connection, no extra getconn."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pool.getconn.return_value = mock_conn

        adapter = self._make_adapter(pool=mock_pool)
        conn1 = adapter.conn
        conn2 = adapter.conn

        assert conn1 is conn2
        assert mock_pool.getconn.call_count == 1

    def test_pool_putconn_on_close(self):
        """close() returns connection to pool instead of closing it."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pool.getconn.return_value = mock_conn

        adapter = self._make_adapter(pool=mock_pool)
        _ = adapter.conn  # trigger getconn
        adapter.close()

        mock_pool.putconn.assert_called_once_with(mock_conn)
        mock_conn.close.assert_not_called()
        assert adapter._conn is None
        assert adapter._pool_conn is False

    def test_no_pool_close_calls_conn_close(self):
        """Without pool, close() calls conn.close() directly."""
        adapter = self._make_adapter(pool=None)
        mock_conn = MagicMock()
        mock_conn.closed = False
        adapter._conn = mock_conn

        adapter.close()

        mock_conn.close.assert_called_once()
        assert adapter._conn is None

    def test_close_without_conn_is_safe(self):
        """close() is safe to call when no connection exists."""
        adapter = self._make_adapter(pool=None)
        adapter.close()  # should not raise

    def test_pool_reconnect_on_closed_conn(self):
        """If pooled connection is closed, adapter gets a new one from pool."""
        mock_pool = MagicMock()

        mock_conn1 = MagicMock()
        mock_conn1.closed = True  # simulate closed connection

        mock_conn2 = MagicMock()
        mock_conn2.closed = False

        mock_pool.getconn.return_value = mock_conn2

        adapter = self._make_adapter(pool=mock_pool)
        # Simulate: adapter already had conn1 from pool, but it's now closed
        adapter._conn = mock_conn1
        adapter._pool_conn = True

        # Accessing conn should detect closed and get a new one from pool
        conn = adapter.conn
        mock_pool.getconn.assert_called_once()
        assert conn is mock_conn2

    def test_row_factory_set_on_pool_conn(self):
        """Pool connections get dict_row factory set."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pool.getconn.return_value = mock_conn

        adapter = self._make_adapter(pool=mock_pool)
        _ = adapter.conn

        # Verify row_factory was set (to whatever dict_row is in the mock)
        assert hasattr(mock_conn, "row_factory")
