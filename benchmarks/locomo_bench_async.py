#!/usr/bin/env python3
"""LoCoMo benchmark — AsyncMemBlock + native Postgres variant.

Mirrors `locomo_bench.py` exactly, but uses:
  - `AsyncMemBlock(storage="postgresql+asyncpg://...")` instead of
    sync `MemBlock(storage="sqlite:///...")`
  - An isolated Postgres schema per benchmark run (auto-dropped on
    cleanup, doesn't pollute `public`)
  - `await mem.store(...)` / `await mem.query(...)`

Same dataset, same metrics, same scoring as the sync version — only
the storage backend changes. Lets us measure whether the native
async + Postgres + pgvector path matches/beats the sync + sqlite +
fastembed local path.

Usage:
    python benchmarks/locomo_bench_async.py /path/to/locomo10.json \\
        --mode hybrid --top-k 10
    python benchmarks/locomo_bench_async.py /path/to/locomo10.json \\
        --mode multihop --top-k 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Ensure memblock is importable from local source
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memblock import AsyncMemBlock, BlockType, SourceType
from memblock.rerankers import (
    BM25Reranker, HeuristicReranker,
)


CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}


# ─── Data loading + metrics (identical to sync version) ──────────────


def load_conversation_sessions(conversation, session_summaries=None):
    sessions = []
    session_num = 1
    while True:
        key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        if key not in conversation:
            break
        dialogs = conversation[key]
        date = conversation.get(date_key, "")
        summary = ""
        if session_summaries:
            summary = session_summaries.get(f"session_{session_num}_summary", "")
        sessions.append({
            "session_num": session_num, "date": date,
            "dialogs": dialogs, "summary": summary,
        })
        session_num += 1
    return sessions


def build_session_text(session):
    return "\n".join(
        f'{d.get("speaker", "?")} said, "{d.get("text", "")}"'
        for d in session["dialogs"]
    )


def evidence_to_session_ids(evidence):
    sessions = set()
    for eid in evidence:
        match = re.match(r"D(\d+):", eid)
        if match:
            sessions.add(f"session_{match.group(1)}")
    return sessions


def compute_retrieval_recall(retrieved_ids, evidence_ids):
    if not evidence_ids:
        return 1.0
    found = sum(1 for eid in evidence_ids if eid in retrieved_ids)
    return found / len(evidence_ids)


# ─── Async ingest + query ────────────────────────────────────────────


async def ingest_sessions(mem: AsyncMemBlock, sessions):
    for sess in sessions:
        doc = build_session_text(sess)
        sess_id = f"session_{sess['session_num']}"
        await mem.store(
            content=doc,
            type=BlockType.FACT,
            source=SourceType.EXPLICIT,
            confidence=1.0,
            session_id=sess_id,
            tags=[sess_id],
            metadata={"corpus_id": sess_id, "timestamp": sess.get("date", "")},
        )


async def query_basic(mem: AsyncMemBlock, question, top_k):
    return await mem.query(
        text_search=question, semantic=True,
        sort_by="relevance", limit=top_k,
    )


async def query_hybrid(mem: AsyncMemBlock, question, top_k):
    # Same as basic — the reranker (when configured at construction)
    # automatically reorders the FTS+vector candidates inside
    # AsyncQueryEngine.
    return await mem.query(
        text_search=question, semantic=True,
        sort_by="relevance", limit=top_k,
    )


async def query_multihop(mem: AsyncMemBlock, question, top_k):
    return await mem.multi_hop_query(query=question, limit=top_k)


def extract_session_ids(blocks):
    retrieved, seen = [], set()
    for block in blocks:
        sid = block.metadata.session_id
        if sid and sid not in seen:
            seen.add(sid)
            retrieved.append(sid)
    return retrieved


# ─── Postgres helpers ────────────────────────────────────────────────


async def drop_schema(dsn: str, schema: str) -> None:
    """Best-effort schema drop — ignores errors."""
    try:
        import asyncpg
        # Strip +asyncpg driver suffix for raw asyncpg.connect
        raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        conn = await asyncpg.connect(raw_dsn)
        try:
            await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await conn.close()
    except Exception as exc:  # noqa: BLE001
        print(f"  [cleanup] schema drop failed: {exc}")


# ─── Benchmark runner ────────────────────────────────────────────────


def _make_reranker(name: str):
    """Pick a reranker by lowercase name. None for no reranker."""
    if not name or name == "none":
        return None
    if name == "heuristic":
        return HeuristicReranker()
    if name == "bm25":
        return BM25Reranker()
    raise ValueError(f"unknown reranker: {name!r}")


async def run_benchmark(
    *,
    data_file: str,
    dsn: str,
    embeddings: str | bool,
    top_k: int = 10,
    mode: str = "basic",
    reranker_name: str = "heuristic",
    limit: int = 0,
    out_file: str | None = None,
):
    with open(data_file) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]

    print(f"\n{'=' * 60}")
    print("  MemBlock × LoCoMo Benchmark — async + Postgres")
    print(f"{'=' * 60}")
    print(f"  Data:          {Path(data_file).name}")
    print(f"  Conversations: {len(data)}")
    print(f"  DSN:           {dsn.split('@')[-1] if '@' in dsn else dsn}")
    print(f"  Top-k:         {top_k}")
    print(f"  Mode:          {mode}")
    print(f"  Embeddings:    {embeddings}")
    print(f"{'─' * 60}\n")

    all_recall = []
    per_category = defaultdict(list)
    results_log = []
    total_qa = 0

    start_time = datetime.now()

    for conv_idx, sample in enumerate(data):
        sample_id = sample.get("sample_id", f"conv-{conv_idx}")
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]
        session_summaries = sample.get("session_summary", {})
        sessions = load_conversation_sessions(conversation, session_summaries)

        # One isolated schema per conversation — exact same isolation
        # semantics as the sync version's tempdb-per-conversation.
        schema = f"locomo_{uuid.uuid4().hex[:10]}"
        print(
            f"  [{conv_idx + 1}/{len(data)}] {sample_id}: "
            f"{len(sessions)} sessions, {len(qa_pairs)} questions  "
            f"(schema={schema})"
        )

        try:
            reranker = _make_reranker(reranker_name) if mode == "hybrid" else None
            mem = AsyncMemBlock(
                storage=dsn,
                embeddings=embeddings,
                reranker=reranker,
                schema=schema,
                user_id=sample_id,
            )

            await ingest_sessions(mem, sessions)

            query_fn = {
                "basic": query_basic,
                "hybrid": query_hybrid,
                "multihop": query_multihop,
            }[mode]

            for qa in qa_pairs:
                question = qa["question"]
                answer = qa.get("answer", qa.get("adversarial_answer", ""))
                category = qa["category"]
                evidence = qa.get("evidence", [])

                result_blocks = await query_fn(mem, question, top_k)
                retrieved_ids = extract_session_ids(result_blocks)
                evidence_ids = evidence_to_session_ids(evidence)
                recall = compute_retrieval_recall(retrieved_ids, evidence_ids)

                all_recall.append(recall)
                per_category[category].append(recall)
                total_qa += 1
                results_log.append({
                    "sample_id": sample_id,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "evidence": list(evidence_ids),
                    "retrieved_ids": retrieved_ids[:top_k],
                    "recall": recall,
                })

            # Close the pool so we don't accumulate connections.
            try:
                await mem.close()
            except Exception:
                pass
        finally:
            await drop_schema(dsn, schema)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'=' * 60}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / total_qa:.2f}s per question)")
    print(f"  Questions:   {total_qa}")
    print(f"  Avg Recall:  {sum(all_recall) / len(all_recall):.3f}")
    print()
    print("  PER-CATEGORY RECALL:")
    for cat in sorted(per_category):
        rs = per_category[cat]
        print(f"    {CATEGORIES.get(cat, str(cat)):<25} R={sum(rs)/len(rs):.3f}  (n={len(rs)})")
    perfect = sum(1 for r in all_recall if r >= 1.0)
    partial = sum(1 for r in all_recall if 0 < r < 1.0)
    zero    = sum(1 for r in all_recall if r == 0)
    print()
    print("  RECALL DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect:>4} ({perfect/len(all_recall)*100:.1f}%)")
    print(f"    Partial (0-1):  {partial:>4} ({partial/len(all_recall)*100:.1f}%)")
    print(f"    Zero (0.0):     {zero:>4} ({zero/len(all_recall)*100:.1f}%)")
    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results_log, f, indent=2, default=str)
        print(f"  Results saved to: {out_file}")


def main():
    p = argparse.ArgumentParser(description="LoCoMo async + Postgres benchmark")
    p.add_argument("data_file", help="Path to locomo10.json")
    p.add_argument(
        "--dsn", default="postgresql+asyncpg://shubham:@localhost:5432/local_qonfido_user",
        help="Postgres DSN (asyncpg-style)",
    )
    p.add_argument(
        "--embeddings", default="true",
        help="Embedding provider: 'true' for local fastembed, 'gemini', etc.",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--mode", default="hybrid", choices=["basic", "hybrid", "multihop"])
    p.add_argument(
        "--reranker", default="heuristic",
        choices=["none", "heuristic", "bm25"],
        help="Reranker (only used when mode=hybrid)",
    )
    p.add_argument("--limit", type=int, default=0, help="N conversations (0 = all)")
    p.add_argument(
        "--out",
        default=str(
            Path(__file__).parent /
            f"results_async_locomo_{datetime.now():%Y%m%d_%H%M%S}.json"
        ),
    )
    args = p.parse_args()

    # Coerce 'true'/'false' to bool when intended
    embeddings = args.embeddings
    if embeddings.lower() in ("true", "1", "yes"):
        embeddings = True
    elif embeddings.lower() in ("false", "0", "no"):
        embeddings = False

    asyncio.run(run_benchmark(
        data_file=args.data_file,
        dsn=args.dsn,
        embeddings=embeddings,
        top_k=args.top_k,
        mode=args.mode,
        reranker_name=args.reranker,
        limit=args.limit,
        out_file=args.out,
    ))


if __name__ == "__main__":
    main()
