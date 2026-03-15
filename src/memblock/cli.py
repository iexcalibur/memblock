"""MemBlock CLI — command-line interface for managing memory databases."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import memblock


def _open_db(db: str) -> memblock.MemBlock:
    """Open a MemBlock database from a URI."""
    return memblock.MemBlock(storage=db)


# ─── Subcommands ──────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> int:
    """Create / initialize a database."""
    mem = _open_db(args.db)
    mem.close()
    print(f"Initialized MemBlock database: {args.db}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Search blocks."""
    mem = _open_db(args.db)
    try:
        block_type = memblock.BlockType(args.type) if args.type else None
        tags = args.tags.split(",") if args.tags else None

        results = mem.query(
            text_search=args.text if args.text else None,
            type=block_type,
            tags=tags,
            limit=args.limit,
        )

        if args.json:
            data = [b.to_dict() for b in results]
            print(json.dumps(data, indent=2, default=str))
        else:
            if not results:
                print("No blocks found.")
            for b in results:
                tags_str = f" [{', '.join(b.tags)}]" if b.tags else ""
                print(f"  {b.id}  [{b.type.value}]  {b.content[:80]}{tags_str}")
        return 0
    finally:
        mem.close()


def cmd_stats(args: argparse.Namespace) -> int:
    """Print memory statistics."""
    mem = _open_db(args.db)
    try:
        s = mem.stats()
        if args.json:
            print(json.dumps(s, indent=2, default=str))
        else:
            print(f"Total blocks:     {s['total_blocks']}")
            print(f"Deleted blocks:   {s['deleted_blocks']}")
            print(f"Total edges:      {s['total_edges']}")
            print(f"Total operations: {s['total_operations']}")
            print(f"Embeddings:       {s['total_embeddings']} ({'enabled' if s['embeddings_enabled'] else 'disabled'})")
            if s["blocks_by_type"]:
                print("Blocks by type:")
                for t, c in sorted(s["blocks_by_type"].items()):
                    print(f"  {t}: {c}")
        return 0
    finally:
        mem.close()


def cmd_prune(args: argparse.Namespace) -> int:
    """Remove weak blocks below a strength threshold."""
    mem = _open_db(args.db)
    try:
        if args.dry_run:
            weak = mem.weakest(limit=1000)
            candidates = [(b, s) for b, s in weak if s < args.min_strength]
            if not candidates:
                print("No blocks below threshold.")
            else:
                print(f"Would prune {len(candidates)} block(s):")
                for b, strength in candidates:
                    print(f"  {b.id}  strength={strength:.4f}  {b.content[:60]}")
        else:
            pruned = mem.prune(min_strength=args.min_strength)
            print(f"Pruned {len(pruned)} block(s).")
        return 0
    finally:
        mem.close()


def cmd_export(args: argparse.Namespace) -> int:
    """Export blocks."""
    mem = _open_db(args.db)
    try:
        if args.format == "json":
            blocks = mem._storage.get_all_blocks()
            data = [b.to_dict() for b in blocks]
            output = json.dumps(data, indent=2, default=str)
        else:
            output = mem.export_markdown()

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Exported {args.format} to {args.output}")
        else:
            print(output)
        return 0
    finally:
        mem.close()


def cmd_reindex(args: argparse.Namespace) -> int:
    """Rebuild the FTS index."""
    mem = _open_db(args.db)
    try:
        # Access the underlying storage to rebuild FTS
        storage = mem._storage
        if not hasattr(storage, "conn"):
            print("Reindex is only supported for SQLite databases.", file=sys.stderr)
            return 1

        conn = storage.conn
        # Rebuild the FTS index
        conn.execute("INSERT INTO blocks_fts(blocks_fts) VALUES('rebuild')")
        conn.commit()
        print("FTS index rebuilt successfully.")
        return 0
    finally:
        mem.close()


def cmd_version(args: argparse.Namespace) -> int:
    """Print version."""
    print(f"memblock {memblock.__version__}")
    return 0


# ─── Parser ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="memblock",
        description="MemBlock CLI — manage AI agent memory databases",
    )
    parser.add_argument(
        "--db",
        default="sqlite:///./memblock.db",
        help="Database URI (default: sqlite:///./memblock.db)",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Create/initialize a database")

    # query
    p_query = sub.add_parser("query", help="Search blocks")
    p_query.add_argument("text", nargs="?", default=None, help="Search text")
    p_query.add_argument("--type", "-t", default=None, help="Filter by block type")
    p_query.add_argument("--tags", default=None, help="Filter by tags (comma-separated)")
    p_query.add_argument("--limit", "-n", type=int, default=10, help="Max results (default: 10)")
    p_query.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # stats
    p_stats = sub.add_parser("stats", help="Print memory statistics")
    p_stats.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # prune
    p_prune = sub.add_parser("prune", help="Remove weak blocks")
    p_prune.add_argument(
        "--min-strength", type=float, default=0.1,
        help="Strength threshold (default: 0.1)",
    )
    p_prune.add_argument("--dry-run", action="store_true", help="Preview without deleting")

    # export
    p_export = sub.add_parser("export", help="Export blocks")
    p_export.add_argument(
        "--format", "-f", choices=["markdown", "json"], default="markdown",
        help="Output format (default: markdown)",
    )
    p_export.add_argument("--output", "-o", default=None, help="Output file path")

    # reindex
    sub.add_parser("reindex", help="Rebuild FTS index")

    # version
    sub.add_parser("version", help="Print version")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init,
        "query": cmd_query,
        "stats": cmd_stats,
        "prune": cmd_prune,
        "export": cmd_export,
        "reindex": cmd_reindex,
        "version": cmd_version,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
