"""Tests for the CLI."""

import json
import os
import tempfile

import pytest

from memblock.cli import main


@pytest.fixture
def tmp_db():
    """Create a temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # MemBlock will create it
    uri = f"sqlite:///{path}"
    yield uri
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def seeded_db(tmp_db):
    """Create a temp DB with some blocks."""
    from memblock import MemBlock, BlockType

    mem = MemBlock(storage=tmp_db)
    mem.store("User prefers Python", type=BlockType.PREFERENCE, tags=["lang"])
    mem.store("User works at Acme", type=BlockType.FACT, tags=["work"])
    mem.store("Met with Alice on Monday", type=BlockType.EVENT)
    mem.close()
    return tmp_db


class TestInit:
    def test_init_creates_db(self, tmp_db):
        rc = main(["--db", tmp_db, "init"])
        assert rc == 0
        # Extract path from URI
        path = tmp_db.replace("sqlite:///", "")
        assert os.path.exists(path)

    def test_init_idempotent(self, tmp_db):
        main(["--db", tmp_db, "init"])
        rc = main(["--db", tmp_db, "init"])
        assert rc == 0


class TestVersion:
    def test_version_output(self, capsys):
        rc = main(["version"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "memblock" in out
        assert "0.4.0" in out


class TestQuery:
    def test_query_text(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "query", "Python"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Python" in out

    def test_query_no_results(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "query", "nonexistent_xyz"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No blocks found" in out

    def test_query_by_type(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "query", "--type", "preference"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "preference" in out

    def test_query_by_tags(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "query", "--tags", "work"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Acme" in out

    def test_query_json(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "query", "Python", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_query_limit(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "query", "--limit", "1"])
        assert rc == 0


class TestStats:
    def test_stats_output(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "stats"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Total blocks:" in out
        assert "3" in out

    def test_stats_json(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "stats", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["total_blocks"] == 3


class TestPrune:
    def test_prune_dry_run(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "prune", "--dry-run", "--min-strength", "999"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Would prune" in out or "No blocks below" in out

    def test_prune_runs(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "prune", "--min-strength", "0.001"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Pruned" in out


class TestExport:
    def test_export_markdown(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "export"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "# MemBlock Export" in out
        assert "Python" in out

    def test_export_json(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "export", "--format", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert len(data) == 3

    def test_export_to_file(self, seeded_db):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            rc = main(["--db", seeded_db, "export", "--output", path])
            assert rc == 0
            with open(path) as f:
                content = f.read()
            assert "# MemBlock Export" in content
        finally:
            os.unlink(path)


class TestReindex:
    def test_reindex(self, seeded_db, capsys):
        rc = main(["--db", seeded_db, "reindex"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "rebuilt" in out.lower()


class TestNoCommand:
    def test_no_command_shows_help(self, capsys):
        rc = main([])
        assert rc == 0

    def test_db_flag(self, tmp_db, capsys):
        rc = main(["--db", tmp_db, "init"])
        assert rc == 0
