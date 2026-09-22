"""Vector fast path: ``brute_force_similarity`` and its use by the
query engines' Python (non-pgvector) fallback.

The numpy path must agree with the pure-Python reference to within
1e-5 and produce the identical ordering; every fallback condition
(numpy missing, big-endian host, mixed-length blobs, dimension
mismatch) must route to the reference loop and still return every row.
"""

from __future__ import annotations

import asyncio
import random
import sys
import warnings
from datetime import timedelta
from unittest.mock import MagicMock

import pytest

import memblock.embeddings as embeddings_mod
from memblock.async_query import AsyncQueryEngine
from memblock.block import Block
from memblock.decay import DecayEngine
from memblock.embeddings import (
    CallableEmbeddingProvider,
    _brute_force_similarity_numpy,
    _brute_force_similarity_python,
    brute_force_similarity,
    cosine_similarity,
    pack_embedding,
    unpack_embedding,
)
from memblock.query import QueryEngine
from memblock.storage.async_base import AsyncStorageAdapter
from memblock.storage.base import StorageAdapter
from memblock.types import BlockMetadata, now_utc

try:
    import numpy  # noqa: F401

    HAS_NUMPY = True
except ImportError:  # pragma: no cover - exercised on numpy-less installs
    HAS_NUMPY = False

requires_numpy = pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")


# ─── Helpers ───────────────────────────────────────────────────────────


def _random_vec(dims: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(dims)]


def _random_embeddings(
    n: int, dims: int, seed: int = 7,
) -> tuple[list[float], list[tuple[str, bytes]]]:
    rng = random.Random(seed)
    query = _random_vec(dims, rng)
    rows = [(f"blk_{i}", pack_embedding(_random_vec(dims, rng))) for i in range(n)]
    return query, rows


def _assert_same_ranking(
    got: list[tuple[str, float]],
    expected: list[tuple[str, float]],
    tol: float = 1e-5,
) -> None:
    assert [bid for bid, _ in got] == [bid for bid, _ in expected]
    for (_, s_got), (_, s_exp) in zip(got, expected):
        assert s_got == pytest.approx(s_exp, abs=tol)


def _spy_python_path(monkeypatch) -> MagicMock:
    """Replace the module-level pure-Python path with a spy that still
    does the real work, so tests can assert which path ran."""
    real = embeddings_mod._brute_force_similarity_python
    spy = MagicMock(side_effect=real)
    monkeypatch.setattr(embeddings_mod, "_brute_force_similarity_python", spy)
    return spy


# ─── numpy vs pure Python ──────────────────────────────────────────────


class TestNumpyMatchesPython:
    @requires_numpy
    def test_random_200x64_scores_and_ordering_agree(self):
        query, rows = _random_embeddings(200, 64)

        fast = _brute_force_similarity_numpy(query, rows)
        slow = _brute_force_similarity_python(query, rows)

        assert fast is not None
        assert len(fast) == len(slow) == 200
        _assert_same_ranking(fast, slow)

    @requires_numpy
    def test_numpy_scores_match_cosine_similarity_directly(self):
        query, rows = _random_embeddings(50, 16, seed=3)
        blobs = dict(rows)
        fast = _brute_force_similarity_numpy(query, rows)
        assert fast is not None
        for bid, score in fast:
            expected = cosine_similarity(query, unpack_embedding(blobs[bid]))
            assert score == pytest.approx(expected, abs=1e-5)

    @requires_numpy
    def test_public_entry_point_takes_numpy_path(self, monkeypatch):
        query, rows = _random_embeddings(20, 8)
        spy = _spy_python_path(monkeypatch)

        result = brute_force_similarity(query, rows)

        spy.assert_not_called()
        assert len(result) == 20
        _assert_same_ranking(result, _brute_force_similarity_python(query, rows))

    def test_results_sorted_descending(self):
        query, rows = _random_embeddings(100, 32)
        result = brute_force_similarity(query, rows)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_returns_plain_python_floats(self):
        query, rows = _random_embeddings(5, 4)
        for bid, score in brute_force_similarity(query, rows):
            assert type(bid) is str
            assert type(score) is float

    def test_ties_keep_input_order_in_both_paths(self):
        vec = [0.3, 0.4, 0.5]
        rows = [("first", pack_embedding(vec)), ("second", pack_embedding(vec)),
                ("third", pack_embedding(vec))]
        query = [0.3, 0.4, 0.5]

        slow = _brute_force_similarity_python(query, rows)
        assert [bid for bid, _ in slow] == ["first", "second", "third"]

        fast = _brute_force_similarity_numpy(query, rows)
        if fast is not None:  # numpy available
            assert [bid for bid, _ in fast] == ["first", "second", "third"]
        assert [bid for bid, _ in brute_force_similarity(query, rows)] == [
            "first", "second", "third",
        ]


# ─── Fallback conditions ───────────────────────────────────────────────


class TestFallbacks:
    def test_empty_input(self):
        assert brute_force_similarity([1.0, 0.0], []) == []

    def test_mixed_dimension_blobs_fall_back_and_return_all_rows(self, monkeypatch):
        rows = [
            ("two", pack_embedding([1.0, 0.0])),
            ("three", pack_embedding([1.0, 0.0, 0.0])),
            ("four", pack_embedding([0.0, 1.0, 0.0, 0.0])),
        ]
        query = [1.0, 0.0]
        spy = _spy_python_path(monkeypatch)

        assert _brute_force_similarity_numpy(query, rows) is None
        result = brute_force_similarity(query, rows)

        spy.assert_called_once()
        assert len(result) == 3
        assert {bid for bid, _ in result} == {"two", "three", "four"}
        # The name imported at module top is the original function object,
        # untouched by the monkeypatch on the module attribute.
        _assert_same_ranking(result, _brute_force_similarity_python(query, rows))

    def test_query_dimension_mismatch_falls_back(self, monkeypatch):
        rows = [("a", pack_embedding([1.0, 0.0, 0.0])), ("b", pack_embedding([0.0, 1.0, 0.0]))]
        query = [1.0, 0.0]  # 2-d query vs 3-d rows
        spy = _spy_python_path(monkeypatch)

        assert _brute_force_similarity_numpy(query, rows) is None
        result = brute_force_similarity(query, rows)

        spy.assert_called_once()
        assert len(result) == 2
        assert result[0][0] == "a"

    def test_zero_vectors_do_not_divide_by_zero(self):
        rows = [
            ("zero_a", pack_embedding([0.0] * 8)),
            ("unit", pack_embedding([1.0] * 8)),
            ("zero_b", pack_embedding([0.0] * 8)),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any RuntimeWarning fails the test

            # zero rows against a real query
            result = brute_force_similarity([1.0] * 8, rows)
            assert [bid for bid, _ in result] == ["unit", "zero_a", "zero_b"]
            assert result[0][1] == pytest.approx(1.0)
            assert result[1][1] == 0.0 and result[2][1] == 0.0

            # zero query against everything
            result = brute_force_similarity([0.0] * 8, rows)
            assert [bid for bid, _ in result] == ["zero_a", "unit", "zero_b"]
            assert all(score == 0.0 for _, score in result)

            # pure-Python reference agrees on both
            _assert_same_ranking(
                brute_force_similarity([1.0] * 8, rows),
                _brute_force_similarity_python([1.0] * 8, rows),
            )

    def test_zero_length_query_falls_back(self, monkeypatch):
        rows = [("a", pack_embedding([1.0, 2.0]))]
        spy = _spy_python_path(monkeypatch)
        assert _brute_force_similarity_numpy([], rows) is None
        result = brute_force_similarity([], rows)
        spy.assert_called_once()
        assert result == [("a", 0.0)]

    @requires_numpy
    def test_numpy_missing_falls_back_identically(self, monkeypatch):
        query, rows = _random_embeddings(200, 64, seed=11)
        with_numpy = brute_force_similarity(query, rows)

        # `sys.modules[name] = None` makes `import numpy` raise ImportError.
        monkeypatch.setitem(sys.modules, "numpy", None)
        spy = _spy_python_path(monkeypatch)

        assert _brute_force_similarity_numpy(query, rows) is None
        without_numpy = brute_force_similarity(query, rows)

        spy.assert_called_once()
        assert len(without_numpy) == 200
        _assert_same_ranking(without_numpy, with_numpy)

    @requires_numpy
    def test_big_endian_host_falls_back(self, monkeypatch):
        query, rows = _random_embeddings(10, 4)
        monkeypatch.setattr(embeddings_mod.sys, "byteorder", "big")
        spy = _spy_python_path(monkeypatch)

        assert _brute_force_similarity_numpy(query, rows) is None
        result = brute_force_similarity(query, rows)
        spy.assert_called_once()
        assert len(result) == 10

    def test_cosine_similarity_public_api_unchanged(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ─── Engine fallback wiring ────────────────────────────────────────────


def _block(block_id: str, minutes_ago: int) -> Block:
    meta = BlockMetadata(
        confidence=0.8, created_at=now_utc() - timedelta(minutes=minutes_ago),
    )
    return Block(id=block_id, content="python", metadata=meta)


def _fallback_fixture():
    """FTS order A, B, C; vectors make B the closest to the query and
    A orthogonal to it. Expected final order: B, C, A."""
    blocks = [_block("A", 3), _block("B", 2), _block("C", 1)]
    packed = [
        ("A", pack_embedding([0.0, 1.0, 0.0])),
        ("B", pack_embedding([1.0, 0.0, 0.0])),
        ("C", pack_embedding([0.9, 0.1, 0.0])),
    ]
    provider = CallableEmbeddingProvider(
        lambda texts: [[1.0, 0.0, 0.0] for _ in texts], dims=3,
    )
    return blocks, packed, provider


class TestEngineFallback:
    def test_sync_engine_ranks_closest_vector_first(self):
        blocks, packed, provider = _fallback_fixture()
        storage = MagicMock(spec=StorageAdapter)
        storage.query_blocks.return_value = blocks
        storage.search_similar_embeddings.return_value = []  # no pgvector
        storage.get_all_embeddings.return_value = packed
        storage.get_block.return_value = None

        engine = QueryEngine(
            storage, graph=MagicMock(),
            decay=DecayEngine(storage=None),  # type: ignore[arg-type]
            embedding_provider=provider,
        )
        ids = [b.id for b in engine.query(text_search="python", limit=10)]

        assert ids[0] == "B"
        assert ids == ["B", "C", "A"]
        storage.search_similar_embeddings.assert_called_once()
        storage.get_all_embeddings.assert_called_once()

    def test_sync_engine_uses_brute_force_similarity(self, monkeypatch):
        blocks, packed, provider = _fallback_fixture()
        storage = MagicMock(spec=StorageAdapter)
        storage.query_blocks.return_value = blocks
        storage.search_similar_embeddings.return_value = []
        storage.get_all_embeddings.return_value = packed
        storage.get_block.return_value = None

        spy = MagicMock(side_effect=embeddings_mod.brute_force_similarity)
        monkeypatch.setattr(embeddings_mod, "brute_force_similarity", spy)

        engine = QueryEngine(
            storage, graph=MagicMock(),
            decay=DecayEngine(storage=None),  # type: ignore[arg-type]
            embedding_provider=provider,
        )
        engine.query(text_search="python", limit=10)

        spy.assert_called_once()
        query_vec, rows = spy.call_args.args
        assert query_vec == [1.0, 0.0, 0.0]
        assert rows == packed

    def test_async_engine_ranks_closest_vector_first(self):
        async def _test():
            blocks, packed, provider = _fallback_fixture()
            storage = MagicMock(spec=AsyncStorageAdapter)
            storage.query_blocks.return_value = blocks
            storage.search_similar_embeddings.return_value = []
            storage.get_all_embeddings.return_value = packed
            storage.get_block.return_value = None

            engine = AsyncQueryEngine(
                storage,
                decay=DecayEngine(storage=None),  # type: ignore[arg-type]
                embedding_provider=provider,
            )
            results = await engine.query(text_search="python", limit=10)
            ids = [b.id for b in results]

            assert ids[0] == "B"
            assert ids == ["B", "C", "A"]
            storage.search_similar_embeddings.assert_called_once()
            storage.get_all_embeddings.assert_called_once()

        asyncio.run(_test())

    def test_async_engine_uses_brute_force_similarity(self, monkeypatch):
        async def _test():
            blocks, packed, provider = _fallback_fixture()
            storage = MagicMock(spec=AsyncStorageAdapter)
            storage.query_blocks.return_value = blocks
            storage.search_similar_embeddings.return_value = []
            storage.get_all_embeddings.return_value = packed
            storage.get_block.return_value = None

            spy = MagicMock(side_effect=embeddings_mod.brute_force_similarity)
            monkeypatch.setattr(embeddings_mod, "brute_force_similarity", spy)

            engine = AsyncQueryEngine(
                storage,
                decay=DecayEngine(storage=None),  # type: ignore[arg-type]
                embedding_provider=provider,
            )
            await engine.query(text_search="python", limit=10)

            spy.assert_called_once()
            query_vec, rows = spy.call_args.args
            assert query_vec == [1.0, 0.0, 0.0]
            assert rows == packed

        asyncio.run(_test())


def test_nan_embeddings_rank_last_on_both_paths():
    """A corrupt (NaN) embedding must sort last identically on the numpy and
    pure-Python paths (review finding: they diverged)."""
    import math

    from memblock.embeddings import (
        _brute_force_similarity_numpy,
        _brute_force_similarity_python,
        pack_embedding,
    )

    rows = [
        ("ok", pack_embedding([1.0, 0.0])),
        ("nan", pack_embedding([math.nan, 1.0])),
        ("ok2", pack_embedding([0.5, 0.5])),
    ]
    py = _brute_force_similarity_python([1.0, 0.0], rows)
    assert [bid for bid, _ in py] == ["ok", "ok2", "nan"]
    np_res = _brute_force_similarity_numpy([1.0, 0.0], rows)
    if np_res is not None:
        assert [bid for bid, _ in np_res] == ["ok", "ok2", "nan"]
