#!/usr/bin/env python3
"""
MemBlock retrieval latency benchmark
====================================

Measures ``MemBlock.query(text_search=...)`` latency on a synthetic SQLite
corpus in FTS-only and hybrid (FTS + vector) modes, and reports a top-1
sample so ranking quality can be checked alongside the timings.

The script imports ``memblock`` from ``<repo>/src`` relative to its own
location, so running the same file from a ``git worktree`` of an older
commit yields BEFORE numbers and running it from the working tree yields
AFTER numbers. The JSON output records the git revision, the resolved
``memblock.__file__`` and the interpreter, so the two runs are attributable.

Corpus
------
Synthetic Indian mutual-fund style sentences: 10 fund names x 5 topics, e.g.

    "Kotak Emerging Equity XIRR return discussion note 483920 reviewed with the advisor"

Every row carries a unique 6-digit token. A "hot" query is ``"<fund> <topic>"``
(e.g. ``"Kotak XIRR"``); memblock OR-joins the words for FTS5, so it matches
every row mentioning the fund (10%) or the topic (20%): ~28% of the corpus.
Only ~2% of rows mention both, so the true best match is easy to identify:
``top1_has_both_terms`` / ``top10_both_terms`` in the report say whether the
ranking surfaces those rows. A "selective" query is one row's 6-digit token
and matches exactly one row.

Hybrid mode
-----------
Uses ``CallableEmbeddingProvider`` (384 dims) with a deterministic token-hash
embedding: each token hashes to a seeded RNG that yields a pseudo-random unit
vector, and a text's vector is the normalised sum of its tokens' vectors.
Deterministic, dependency-free (numpy is used only when importable, for
speed), and a query that shares tokens with a document is genuinely closer to
it, so the hybrid top-1 sample is meaningful rather than noise.

Usage
-----
    python benchmarks/latency_bench.py --n 1000  --mode both
    python benchmarks/latency_bench.py --n 10000 --mode fts    --json fts_10k.json
    python benchmarks/latency_bench.py --n 10000 --mode hybrid --queries 20

At n=10000 the hybrid query count is capped at 30 by default (``--hybrid-queries``)
because the pure-Python cosine fallback costs ~0.4 s per query on the pre-fix code.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import memblock  # noqa: E402
from memblock import BlockType, MemBlock  # noqa: E402
from memblock.embeddings import CallableEmbeddingProvider  # noqa: E402

DIMS = 384
SELECTIVE_QUERIES = 20
HYBRID_DEFAULT_CAP = 30

# (query token, full fund name). The query token is unique to its fund.
FUNDS = [
    ("Kotak", "Kotak Emerging Equity"),
    ("Parikh", "Parag Parikh Flexi Cap"),
    ("Mirae", "Mirae Asset Large Cap"),
    ("Axis", "Axis Bluechip"),
    ("SBI", "SBI Small Cap"),
    ("HDFC", "HDFC Mid Cap Opportunities"),
    ("ICICI", "ICICI Prudential Technology"),
    ("Nippon", "Nippon India Growth"),
    ("UTI", "UTI Nifty Index"),
    ("Quant", "Quant Active"),
]

# (query token, topic phrase). The query token is unique to its topic.
TOPICS = [
    ("XIRR", "XIRR return"),
    ("SIP", "SIP installment"),
    ("NAV", "NAV movement"),
    ("drawdown", "drawdown risk"),
    ("rebalancing", "rebalancing allocation"),
]

# Filler phrases add variety; none of their words collide with a query token
# or with a temporal-parser trigger ("last", "before", "after", "recent", ...).
FILLERS = [
    "reviewed with the advisor",
    "flagged during the quarterly portfolio check",
    "noted from the CAMS statement",
    "captured from the app dashboard",
    "shared by the relationship manager",
    "compared against the benchmark",
]


# ── corpus and queries ──────────────────────────────────────────────────────


def build_corpus(n: int, rng: random.Random) -> list[dict]:
    """n rows; each fund gets exactly ~10% and each topic ~20% of rows."""
    combos = [(i % len(FUNDS), (i // len(FUNDS)) % len(TOPICS)) for i in range(n)]
    rng.shuffle(combos)
    tokens = rng.sample(range(100000, 1000000), n)
    rows = []
    for (fi, ti), tok in zip(combos, tokens):
        fund_key, fund_name = FUNDS[fi]
        topic_key, topic_phrase = TOPICS[ti]
        filler = FILLERS[rng.randrange(len(FILLERS))]
        rows.append(
            {
                "content": f"{fund_name} {topic_phrase} discussion note {tok} {filler}",
                "fund": fund_key,
                "topic": topic_key,
                "token": tok,
            }
        )
    return rows


def build_hot_queries(n_hot: int) -> list[tuple[str, str, str]]:
    """Cycle through the 50 fund x topic combos: (query, fund_key, topic_key)."""
    combos = [(f, t) for f, _ in FUNDS for t, _ in TOPICS]
    out = []
    for i in range(n_hot):
        f, t = combos[i % len(combos)]
        out.append((f"{f} {t}", f, t))
    return out


def build_selective_queries(rows: list[dict], rng: random.Random) -> list[tuple[str, int]]:
    picked = rng.sample(rows, min(SELECTIVE_QUERIES, len(rows)))
    return [(str(r["token"]), r["token"]) for r in picked]


# ── deterministic embeddings ───────────────────────────────────────────────


def make_embed_fn(dims: int):
    """Token-hash embedding: sha256(token) -> seeded RNG -> unit vector; text = normalised sum."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - exercised only without numpy
        np = None

    cache: dict[str, object] = {}

    def token_vec(tok: str):
        vec = cache.get(tok)
        if vec is not None:
            return vec
        seed = int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:8], "little")
        if np is not None:
            vec = np.random.default_rng(seed).standard_normal(dims)
        else:
            r = random.Random(seed)
            vec = [r.gauss(0.0, 1.0) for _ in range(dims)]
        if not tok.isdigit():  # the 6-digit row tokens are one-shot; do not cache them
            cache[tok] = vec
        return vec

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            toks = re.findall(r"\w+", text.lower())
            if np is not None:
                acc = np.zeros(dims)
                for t in toks:
                    acc += token_vec(t)
                norm = float(np.linalg.norm(acc)) or 1.0
                out.append((acc / norm).tolist())
            else:
                acc_l = [0.0] * dims
                for t in toks:
                    tv = token_vec(t)
                    for i in range(dims):
                        acc_l[i] += tv[i]
                norm = math.sqrt(sum(x * x for x in acc_l)) or 1.0
                out.append([x / norm for x in acc_l])
        return out

    return embed


# ── helpers ────────────────────────────────────────────────────────────────


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(p / 100.0 * len(s)) - 1))
    return s[idx]


def summarize(lat_ms: list[float]) -> dict:
    if not lat_ms:
        return {"queries": 0}
    return {
        "queries": len(lat_ms),
        "p50_ms": round(statistics.median(lat_ms), 3),
        "p95_ms": round(percentile(lat_ms, 95), 3),
        "mean_ms": round(statistics.fmean(lat_ms), 3),
        "max_ms": round(max(lat_ms), 3),
    }


def git_info(repo: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    rev = run("rev-parse", "--short", "HEAD")
    dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"rev": rev, "dirty": bool(dirty) if dirty is not None else None}


def mode_db_path(base: Path, mode: str) -> Path:
    return base.with_name(f"{base.stem}-{mode}{base.suffix or '.sqlite'}")


def open_memblock(db_path: Path, provider: CallableEmbeddingProvider | None) -> MemBlock:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()
    # sqlite:/// + absolute path -> sqlite:////abs/path, parsed as /abs/path.
    mem = MemBlock(storage=f"sqlite:///{db_path}", embeddings=False)
    if provider is not None:
        # Same injection the test-suite uses; embeddings=True would need fastembed.
        mem._embedding_provider = provider
        mem._query._embedding_provider = provider
    return mem


def fts_match_rows(mem: MemBlock, query: str) -> int | None:
    """How many rows the OR-joined FTS5 query matches (same sanitising as the adapter)."""
    words = re.findall(r"\w+", query)
    fts_query = " OR ".join(f'"{w}"' for w in words) if words else '""'
    try:
        conn = mem._storage.conn
        row = conn.execute(
            "SELECT count(*) FROM blocks_fts WHERE blocks_fts MATCH ?", (fts_query,)
        ).fetchone()
        return int(row[0])
    except Exception:
        return None


def has_terms(content: str, *terms: str) -> bool:
    words = set(re.findall(r"\w+", content.lower()))
    return all(t.lower() in words for t in terms)


def time_queries(mem: MemBlock, queries: list[str], limit: int, semantic: bool):
    lat: list[float] = []
    results = []
    for q in queries:
        t0 = time.perf_counter()
        res = mem.query(text_search=q, limit=limit, semantic=semantic)
        lat.append((time.perf_counter() - t0) * 1000.0)
        results.append(res)
    return lat, results


# ── one mode ───────────────────────────────────────────────────────────────


def run_mode(
    mode: str,
    rows: list[dict],
    hot: list[tuple[str, str, str]],
    selective: list[tuple[str, int]],
    args: argparse.Namespace,
) -> dict:
    semantic = mode == "hybrid"
    provider = CallableEmbeddingProvider(make_embed_fn(DIMS), DIMS) if semantic else None
    if semantic:
        hot = hot[: min(len(hot), args.hybrid_queries)]

    db_path = mode_db_path(args.db, mode)
    print(f"\n[{mode}] db={db_path}")
    mem = open_memblock(db_path, provider)
    try:
        # store
        t0 = time.perf_counter()
        for r in rows:
            mem.store(
                r["content"],
                type=BlockType.FACT,
                tags=[r["fund"].lower(), r["topic"].lower()],
            )
        store_s = time.perf_counter() - t0
        print(f"[{mode}] stored {len(rows)} blocks in {store_s:.2f}s "
              f"({len(rows) / store_s:.0f} blocks/s)")

        # warm-up (untimed)
        for q, _, _ in hot[:2]:
            mem.query(text_search=q, limit=args.limit, semantic=semantic)

        hot_lat, hot_res = time_queries(mem, [q for q, _, _ in hot], args.limit, semantic)
        sel_lat, sel_res = time_queries(mem, [q for q, _ in selective], args.limit, semantic)

        # ranking quality over the hot set: how many of the top-k mention BOTH terms
        both_at_k = []
        top1_both = 0
        for (q, f, t), res in zip(hot, hot_res):
            both_at_k.append(sum(1 for b in res if has_terms(b.content, f, t)))
            if res and has_terms(res[0].content, f, t):
                top1_both += 1

        # selective: the single matching row must come back first
        sel_hits = sum(
            1 for (q, tok), res in zip(selective, sel_res)
            if res and str(tok) in res[0].content
        )

        q0, f0, t0k = hot[0]
        res0 = hot_res[0]
        n_match = fts_match_rows(mem, q0)
        sample = {
            "query": q0,
            "fts_match_rows": n_match,
            "fts_match_pct": round(100.0 * n_match / len(rows), 1) if n_match is not None else None,
            "rows_with_both_terms": sum(1 for r in rows if r["fund"] == f0 and r["topic"] == t0k),
            "top1": res0[0].content if res0 else None,
            "top1_has_both_terms": bool(res0) and has_terms(res0[0].content, f0, t0k),
            f"top{args.limit}_both_terms": both_at_k[0] if both_at_k else 0,
            "latency_ms": round(hot_lat[0], 3) if hot_lat else None,
        }

        report = {
            "n": len(rows),
            "limit": args.limit,
            "semantic": semantic,
            "embeddings_stored": len(mem._storage.get_all_embeddings()) if semantic else 0,
            "store_seconds": round(store_s, 3),
            "store_blocks_per_sec": round(len(rows) / store_s, 1),
            "hot": {
                **summarize(hot_lat),
                "top1_has_both_terms_rate": round(top1_both / len(hot), 3) if hot else None,
                f"mean_top{args.limit}_both_terms": round(statistics.fmean(both_at_k), 2) if both_at_k else None,
            },
            "selective": {
                **summarize(sel_lat),
                "top1_hits": sel_hits,
            },
            "all": summarize(hot_lat + sel_lat),
            "hot_sample": sample,
        }
    finally:
        mem.close()

    h, s = report["hot"], report["selective"]
    print(f"[{mode}] hot       n={h['queries']:<4} P50={h['p50_ms']:.2f}ms  P95={h['p95_ms']:.2f}ms  "
          f"top1-both={h['top1_has_both_terms_rate']}  mean-top{args.limit}-both={h[f'mean_top{args.limit}_both_terms']}")
    print(f"[{mode}] selective n={s['queries']:<4} P50={s['p50_ms']:.2f}ms  P95={s['p95_ms']:.2f}ms  "
          f"top1-hits={s['top1_hits']}/{s['queries']}")
    print(f"[{mode}] sample    {sample['query']!r} matches {sample['fts_match_rows']} rows "
          f"({sample['fts_match_pct']}%), {sample['rows_with_both_terms']} have both terms")
    print(f"[{mode}]           top-1: {sample['top1']!r}  both-terms={sample['top1_has_both_terms']}")
    return report


# ── main ───────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    scratch = Path(tempfile.gettempdir())
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10000, help="blocks to store (default 10000)")
    p.add_argument("--queries", type=int, default=100, help="hot queries per mode (default 100)")
    p.add_argument("--hybrid-queries", type=int, default=None,
                   help=f"cap on hot queries in hybrid mode (default min(--queries, {HYBRID_DEFAULT_CAP}))")
    p.add_argument("--mode", choices=["fts", "hybrid", "both"], default="both")
    p.add_argument("--db", type=Path, default=scratch / "latency_bench.sqlite",
                   help="SQLite base path; '-<mode>' is inserted before the suffix (default under the system temp dir)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=10, help="query limit (default 10)")
    p.add_argument("--json", type=Path, default=None, help="write results JSON here")
    args = p.parse_args(argv)
    if args.hybrid_queries is None:
        args.hybrid_queries = min(args.queries, HYBRID_DEFAULT_CAP)
    if args.n < 1:
        p.error("--n must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    faulthandler.dump_traceback_later(600, exit=True)
    try:
        args = parse_args(argv)
        rng = random.Random(args.seed)
        rows = build_corpus(args.n, rng)
        hot = build_hot_queries(args.queries)
        selective = build_selective_queries(rows, rng)

        try:
            import numpy as np
            numpy_version = np.__version__
        except ImportError:
            numpy_version = None

        meta = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo": str(REPO),
            "git": git_info(REPO),
            "memblock_file": memblock.__file__,
            "memblock_version": getattr(memblock, "__version__", None),
            "python": sys.version.split()[0],
            "numpy": numpy_version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        print(f"memblock {meta['memblock_version']} @ {meta['git']['rev']}"
              f"{' (dirty)' if meta['git']['dirty'] else ''}  from {meta['memblock_file']}")
        print(f"python {meta['python']}  numpy {numpy_version}  n={args.n}  seed={args.seed}")

        modes = ["fts", "hybrid"] if args.mode == "both" else [args.mode]
        results = {"meta": meta, "modes": {}}
        for mode in modes:
            results["modes"][mode] = run_mode(mode, rows, hot, selective, args)

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(results, indent=2))
            print(f"\nwrote {args.json}")
        return 0
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())
