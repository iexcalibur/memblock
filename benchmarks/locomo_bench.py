#!/usr/bin/env python3
"""
MemBlock × LoCoMo Benchmark
=============================

Evaluates MemBlock's retrieval against the LoCoMo benchmark.
10 conversations, ~200 QA pairs across 5 categories.

Uses the same dataset, metrics, and evaluation methodology as
MemPalace's locomo_bench.py for direct comparison.

For each conversation:
1. Ingest all sessions into a fresh MemBlock instance
2. For each QA pair, query MemBlock
3. Score retrieval recall (did we find the evidence session?)

Usage:
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json --mode multihop
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json --mode graph
    python benchmarks/locomo_bench.py /path/to/locomo/data/locomo10.json --top-k 10 --limit 2
"""

import os
import sys
import json
import re
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Ensure memblock is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memblock import MemBlock, BlockType, SourceType
from memblock.rerankers import HeuristicReranker


# ── LoCoMo categories ───────────────────────────────────────────────────────

CATEGORIES = {
    1: "Single-hop",
    2: "Temporal",
    3: "Temporal-inference",
    4: "Open-domain",
    5: "Adversarial",
}


# =============================================================================
# DATA LOADING (mirrors MemPalace's locomo_bench.py exactly)
# =============================================================================


def load_conversation_sessions(conversation, session_summaries=None):
    """Extract sessions from a LoCoMo conversation dict."""
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
        sessions.append(
            {
                "session_num": session_num,
                "date": date,
                "dialogs": dialogs,
                "summary": summary,
            }
        )
        session_num += 1
    return sessions


def build_session_text(session):
    """Build document text from a session (same format as MemPalace)."""
    texts = []
    for d in session["dialogs"]:
        speaker = d.get("speaker", "?")
        text = d.get("text", "")
        texts.append(f'{speaker} said, "{text}"')
    return "\n".join(texts)


# =============================================================================
# METRICS (identical to MemPalace's locomo_bench.py)
# =============================================================================


def compute_retrieval_recall(retrieved_ids, evidence_ids):
    """What fraction of evidence session IDs were retrieved?"""
    if not evidence_ids:
        return 1.0
    found = sum(1 for eid in evidence_ids if eid in retrieved_ids)
    return found / len(evidence_ids)


def evidence_to_session_ids(evidence):
    """Convert LoCoMo evidence (D1:5 format) to session IDs (session_1 format)."""
    sessions = set()
    for eid in evidence:
        match = re.match(r"D(\d+):", eid)
        if match:
            sessions.add(f"session_{match.group(1)}")
    return sessions


# =============================================================================
# MEMBLOCK INGEST + QUERY
# =============================================================================


def ingest_sessions(mem, sessions):
    """Store each session as a FACT block in MemBlock."""
    blocks = []
    for sess in sessions:
        doc = build_session_text(sess)
        sess_id = f"session_{sess['session_num']}"
        block = mem.store(
            content=doc,
            type=BlockType.FACT,
            source=SourceType.EXPLICIT,
            confidence=1.0,
            session_id=sess_id,
            tags=[sess_id],
            metadata={"corpus_id": sess_id, "timestamp": sess.get("date", "")},
        )
        blocks.append((sess_id, block))
    return blocks


def query_basic(mem, question, top_k):
    """Basic hybrid query: FTS5 + vector embeddings via RRF."""
    results = mem.query(
        text_search=question,
        semantic=True,
        sort_by="relevance",
        limit=top_k,
    )
    return results


def query_multihop(mem, question, top_k):
    """Multi-hop iterative retrieval with entity extraction + graph walk."""
    results = mem.multi_hop_query(
        text_search=question,
        limit=top_k,
    )
    return results


def query_hybrid(mem, question, top_k):
    """Hybrid query with heuristic reranking (keyword, phrase, name boosting)."""
    results = mem.query(
        text_search=question,
        semantic=True,
        sort_by="relevance",
        limit=top_k,
    )
    return results


def query_graph(mem, question, top_k):
    """Hybrid query on a graph-linked store (auto-link enabled at ingest)."""
    results = mem.query(
        text_search=question,
        semantic=True,
        sort_by="relevance",
        limit=top_k,
    )
    return results


def extract_session_ids(blocks):
    """Extract session IDs from returned blocks."""
    retrieved = []
    seen = set()
    for block in blocks:
        sid = block.metadata.session_id
        if sid and sid not in seen:
            seen.add(sid)
            retrieved.append(sid)
    return retrieved


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def run_benchmark(
    data_file,
    top_k=50,
    mode="basic",
    limit=0,
    out_file=None,
):
    """Run LoCoMo retrieval benchmark using MemBlock."""
    with open(data_file) as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    print(f"\n{'=' * 60}")
    print("  MemBlock × LoCoMo Benchmark")
    print(f"{'=' * 60}")
    print(f"  Data:          {Path(data_file).name}")
    print(f"  Conversations: {len(data)}")
    print(f"  Top-k:         {top_k}")
    print(f"  Mode:          {mode}")
    print(f"  Granularity:   session")
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

        print(
            f"  [{conv_idx + 1}/{len(data)}] {sample_id}: "
            f"{len(sessions)} sessions, {len(qa_pairs)} questions"
        )

        # Fresh MemBlock per conversation (isolation, same as MemPalace's temp ChromaDB)
        tmpdb = tempfile.mktemp(suffix=".db", prefix="memblock_locomo_")
        try:
            reranker = HeuristicReranker() if mode == "hybrid" else None
            mem = MemBlock(
                storage=f"sqlite:///{tmpdb}",
                embeddings=True,
                reranker=reranker,
            )

            # Graph mode: enable auto-linking before ingest
            if mode == "graph":
                mem.enable_auto_link(enabled=True, max_neighbors=5)

            # Ingest all sessions
            ingest_sessions(mem, sessions)

            # Dispatch query function
            query_fn = {
                "basic": query_basic,
                "hybrid": query_hybrid,
                "multihop": query_multihop,
                "graph": query_graph,
            }[mode]

            for qa in qa_pairs:
                question = qa["question"]
                answer = qa.get("answer", qa.get("adversarial_answer", ""))
                category = qa["category"]
                evidence = qa.get("evidence", [])

                # Query MemBlock
                result_blocks = query_fn(mem, question, top_k)

                # Extract session IDs from results
                retrieved_ids = extract_session_ids(result_blocks)

                # Convert evidence to session IDs
                evidence_set = evidence_to_session_ids(evidence)

                # Compute recall
                recall = compute_retrieval_recall(retrieved_ids, evidence_set)
                all_recall.append(recall)
                per_category[category].append(recall)
                total_qa += 1

                results_log.append(
                    {
                        "sample_id": sample_id,
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "evidence": evidence,
                        "retrieved_ids": retrieved_ids,
                        "recall": recall,
                    }
                )

            mem.close()
        finally:
            # Cleanup temp database
            try:
                os.unlink(tmpdb)
            except OSError:
                pass

    elapsed = (datetime.now() - start_time).total_seconds()

    # ── Print results ────────────────────────────────────────────────────────

    avg_recall = sum(all_recall) / len(all_recall) if all_recall else 0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — MemBlock ({mode}, session, top-{top_k})")
    print(f"{'=' * 60}")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / max(total_qa, 1):.2f}s per question)")
    print(f"  Questions:   {total_qa}")
    print(f"  Avg Recall:  {avg_recall:.3f}")

    print("\n  PER-CATEGORY RECALL:")
    for cat in sorted(per_category.keys()):
        vals = per_category[cat]
        avg = sum(vals) / len(vals)
        name = CATEGORIES.get(cat, f"Cat-{cat}")
        print(f"    {name:25} R={avg:.3f}  (n={len(vals)})")

    perfect = sum(1 for r in all_recall if r >= 1.0)
    partial = sum(1 for r in all_recall if 0 < r < 1.0)
    zero = sum(1 for r in all_recall if r == 0)
    print("\n  RECALL DISTRIBUTION:")
    print(f"    Perfect (1.0):  {perfect:4} ({perfect / len(all_recall) * 100:.1f}%)")
    print(f"    Partial (0-1):  {partial:4} ({partial / len(all_recall) * 100:.1f}%)")
    print(f"    Zero (0.0):     {zero:4} ({zero / len(all_recall) * 100:.1f}%)")

    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"  Results saved to: {out_file}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemBlock × LoCoMo Benchmark")
    parser.add_argument("data_file", help="Path to locomo10.json")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k retrieval (default: 50)")
    parser.add_argument(
        "--mode",
        choices=["basic", "hybrid", "multihop", "graph"],
        default="basic",
        help="Retrieval mode: basic (FTS+vector), hybrid (+ keyword/phrase/name reranking), multihop (entity extraction + graph walk), graph (auto-linked store)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N conversations (0 = all)")
    parser.add_argument("--out", default=None, help="Output JSON file path")

    args = parser.parse_args()

    if not args.out:
        args.out = (
            f"benchmarks/results_memblock_locomo_{args.mode}"
            f"_top{args.top_k}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )

    run_benchmark(
        args.data_file,
        args.top_k,
        args.mode,
        args.limit,
        args.out,
    )
