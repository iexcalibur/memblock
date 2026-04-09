#!/usr/bin/env python3
"""
MemBlock × LongMemEval Benchmark
==================================

Evaluates MemBlock's retrieval against LongMemEval (500 questions, ~53 sessions each).
Uses the same dataset, metrics, and methodology as MemPalace's longmemeval_bench.py.

For each question:
1. Build a haystack of ~53 conversation sessions
2. Index all sessions into a fresh MemBlock instance
3. Query with the question
4. Score: Recall@5, Recall@10, NDCG@10

Usage:
    python benchmarks/longmemeval_bench.py /path/to/longmemeval_s_cleaned.json
    python benchmarks/longmemeval_bench.py /path/to/longmemeval_s_cleaned.json --mode hybrid
    python benchmarks/longmemeval_bench.py /path/to/longmemeval_s_cleaned.json --limit 50
"""

import os
import sys
import json
import math
import argparse
import tempfile
from pathlib import Path
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memblock import MemBlock, BlockType, SourceType
from memblock.rerankers import HeuristicReranker


# =============================================================================
# METRICS (identical to MemPalace's longmemeval_bench.py)
# =============================================================================


def dcg(relevances, k):
    """Discounted Cumulative Gain at rank k."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_score(rankings, correct_ids, corpus_ids, k):
    """Normalized DCG at rank k."""
    relevances = [1.0 if corpus_ids[idx] in correct_ids else 0.0 for idx in rankings[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def evaluate_retrieval(rankings, correct_ids, corpus_ids, k):
    """Compute recall_any@k, recall_all@k, and ndcg@k."""
    top_k_ids = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(cid in top_k_ids for cid in correct_ids))
    recall_all = float(all(cid in top_k_ids for cid in correct_ids))
    nd = ndcg_score(rankings, correct_ids, corpus_ids, k)
    return recall_any, recall_all, nd


def session_id_from_corpus_id(cid):
    """Extract session ID from corpus ID (handles turn-level IDs)."""
    if "_turn_" in cid:
        return cid.rsplit("_turn_", 1)[0]
    return cid


# =============================================================================
# CORPUS BUILDING
# =============================================================================


def build_corpus(entry, granularity="session"):
    """
    Build retrieval corpus from haystack sessions.

    granularity:
        'session' — one doc per session (user turns joined)
        'turn'    — one doc per user turn
    """
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry.get("haystack_dates", [""] * len(sessions))

    corpus = []
    corpus_ids = []
    corpus_timestamps = []

    for sess, sid, date in zip(sessions, session_ids, dates):
        if granularity == "session":
            user_turns = [t["content"] for t in sess if t.get("role") == "user"]
            doc = "\n".join(user_turns)
            if doc.strip():
                corpus.append(doc)
                corpus_ids.append(sid)
                corpus_timestamps.append(date)
        else:  # turn
            for turn_num, t in enumerate(sess):
                if t.get("role") == "user":
                    doc = t["content"]
                    if doc.strip():
                        corpus.append(doc)
                        corpus_ids.append(f"{sid}_turn_{turn_num}")
                        corpus_timestamps.append(date)

    return corpus, corpus_ids, corpus_timestamps


# =============================================================================
# MEMBLOCK INGEST + RETRIEVE
# =============================================================================


def ingest_and_retrieve(mem, corpus, corpus_ids, corpus_timestamps, question, top_n=50):
    """
    Store corpus into MemBlock, query, and return ranked indices.

    Returns:
        rankings: list of indices into corpus, sorted by relevance
    """
    # Ingest all documents
    block_id_to_idx = {}
    for i, (doc, cid, ts) in enumerate(zip(corpus, corpus_ids, corpus_timestamps)):
        block = mem.store(
            content=doc,
            type=BlockType.FACT,
            source=SourceType.EXPLICIT,
            confidence=1.0,
            session_id=cid,
            tags=[cid],
            metadata={"corpus_id": cid, "timestamp": ts},
        )
        block_id_to_idx[block.id] = i

    # Query
    results = mem.query(
        text_search=question,
        semantic=True,
        sort_by="relevance",
        limit=min(top_n, len(corpus)),
    )

    # Map results back to corpus indices
    rankings = []
    seen = set()
    for block in results:
        idx = block_id_to_idx.get(block.id)
        if idx is not None and idx not in seen:
            seen.add(idx)
            rankings.append(idx)

    # Fill in missing indices (unretrieved docs get worst ranks)
    for i in range(len(corpus)):
        if i not in seen:
            rankings.append(i)

    return rankings


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def run_benchmark(
    data_file,
    mode="basic",
    granularity="session",
    limit=0,
    out_file=None,
):
    """Run LongMemEval retrieval benchmark using MemBlock."""
    with open(data_file) as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    print(f"\n{'=' * 60}")
    print("  MemBlock × LongMemEval Benchmark")
    print(f"{'=' * 60}")
    print(f"  Data:          {Path(data_file).name}")
    print(f"  Questions:     {len(data)}")
    print(f"  Mode:          {mode}")
    print(f"  Granularity:   {granularity}")
    print(f"{'─' * 60}\n")

    ks = [1, 3, 5, 10, 30, 50]
    metrics = {f"{m}@{k}": [] for k in ks for m in ["recall_any", "recall_all", "ndcg"]}
    per_type = defaultdict(lambda: defaultdict(list))
    results_log = []

    start_time = datetime.now()

    for i, entry in enumerate(data):
        qid = entry["question_id"]
        qtype = entry["question_type"]
        question = entry["question"]
        correct_ids = set(entry["answer_session_ids"])

        # Build corpus
        corpus, corpus_ids, corpus_timestamps = build_corpus(entry, granularity)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{len(data)}] {len(corpus)} docs, type={qtype}")

        # Fresh MemBlock per question
        tmpdb = tempfile.mktemp(suffix=".db", prefix="memblock_lme_")
        try:
            reranker = HeuristicReranker() if mode == "hybrid" else None
            mem = MemBlock(
                storage=f"sqlite:///{tmpdb}",
                embeddings=True,
                reranker=reranker,
            )

            rankings = ingest_and_retrieve(
                mem, corpus, corpus_ids, corpus_timestamps, question
            )

            # For session-level evaluation, map corpus_ids to session IDs
            session_ids = [session_id_from_corpus_id(cid) for cid in corpus_ids]

            for k in ks:
                ra, rl, nd = evaluate_retrieval(rankings, correct_ids, session_ids, k)
                metrics[f"recall_any@{k}"].append(ra)
                metrics[f"recall_all@{k}"].append(rl)
                metrics[f"ndcg@{k}"].append(nd)

            # Per-type tracking
            per_type[qtype]["recall_any@5"].append(metrics["recall_any@5"][-1])
            per_type[qtype]["recall_any@10"].append(metrics["recall_any@10"][-1])
            per_type[qtype]["ndcg@10"].append(metrics["ndcg@10"][-1])

            results_log.append({
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "recall_any@5": metrics["recall_any@5"][-1],
                "recall_any@10": metrics["recall_any@10"][-1],
                "ndcg@10": metrics["ndcg@10"][-1],
            })

            mem.close()
        finally:
            try:
                os.unlink(tmpdb)
            except OSError:
                pass

    elapsed = (datetime.now() - start_time).total_seconds()

    # ── Print results ────────────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print(f"  RESULTS — MemBlock ({mode}, {granularity})")
    print(f"{'=' * 60}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed / max(len(data), 1):.2f}s per question)")
    print(f"  Questions: {len(data)}")

    print(f"\n  SESSION-LEVEL METRICS:")
    for k in ks:
        ra = sum(metrics[f"recall_any@{k}"]) / len(metrics[f"recall_any@{k}"])
        nd = sum(metrics[f"ndcg@{k}"]) / len(metrics[f"ndcg@{k}"])
        print(f"    Recall@{k:<2}: {ra:.3f}    NDCG@{k:<2}: {nd:.3f}")

    print(f"\n  PER-TYPE BREAKDOWN (Recall@5 / Recall@10):")
    for qtype in sorted(per_type.keys()):
        vals = per_type[qtype]
        r5 = sum(vals["recall_any@5"]) / len(vals["recall_any@5"])
        r10 = sum(vals["recall_any@10"]) / len(vals["recall_any@10"])
        n = len(vals["recall_any@5"])
        print(f"    {qtype:40} R@5={r5:.3f}  R@10={r10:.3f}  (n={n})")

    print(f"\n{'=' * 60}\n")

    if out_file:
        with open(out_file, "w") as f:
            for entry in results_log:
                f.write(json.dumps(entry) + "\n")
        print(f"  Results saved to: {out_file}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemBlock × LongMemEval Benchmark")
    parser.add_argument("data_file", help="Path to longmemeval_s_cleaned.json")
    parser.add_argument(
        "--mode",
        choices=["basic", "hybrid"],
        default="basic",
        help="Retrieval mode: basic (FTS+vector), hybrid (+ heuristic reranking)",
    )
    parser.add_argument(
        "--granularity",
        choices=["session", "turn"],
        default="session",
        help="Corpus granularity (default: session)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit to N questions (0 = all 500)")
    parser.add_argument("--out", default=None, help="Output JSONL file path")

    args = parser.parse_args()

    if not args.out:
        args.out = (
            f"benchmarks/results_memblock_lme_{args.mode}"
            f"_{args.granularity}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        )

    run_benchmark(
        args.data_file,
        args.mode,
        args.granularity,
        args.limit,
        args.out,
    )
