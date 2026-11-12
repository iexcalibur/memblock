#!/usr/bin/env python3
"""
Evaluation of the session layer (MemorySession, transactions, receipts).

Nothing here calls an LLM. By default retrieval is keyword search (FTS) only
and no model is loaded at all. `--embeddings` switches on memblock's default
local embedding model (pip install memblock[embeddings]) so search is keyword
plus vector; `--rerank` adds the heuristic reranker on top.

    python benchmarks/session_bench.py lme    DATA.json [--limit N] [--embeddings] [--rerank]
    python benchmarks/session_bench.py noise  DATA.json [--limit N] [--embeddings] [--rerank]
    python benchmarks/session_bench.py growth DATA.json
    python benchmarks/session_bench.py crash  [--trials N]

DATA.json is LongMemEval's longmemeval_s_cleaned.json.

`--question-clock` reads a time phrase in a question ("last month") as of the
question's own date, as live use would. Without it the phrase is read as of
today; the data is from 2023, so the time window then matches nothing and
memblock ignores it.

With --embeddings, `--embedding-cache FILE.npz` keeps the vectors between runs
and `--workers N` embeds in N processes. The full file has about 208,000
distinct texts to embed: several minutes with 10 workers, and nothing on a
second run with the same cache.

lme     Does a conversation that went through the session layer stay findable?
        Each question's history (about 490 messages) is replayed turn by turn
        through MemorySession, which commits every turn and drops it from the
        prompt. Then the question is asked. Reported against the existing
        benchmark's method (one block per session, queried directly) under the
        same retrieval settings.
noise   Does extra, unrelated memory push the evidence out? Re-measures after
        adding other users' conversations to the same store.
growth  Prompt size per turn: the session layer against sending the whole
        transcript.
crash   Kills a commit at a random point, reopens the store, recovers, and
        checks that a turn is saved completely or not at all, and never twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memblock import BlockType, MemBlock, MemorySession, SourceType  # noqa: E402
from memblock.embeddings import EmbeddingProvider  # noqa: E402
from memblock.transactions import is_internal, is_staged  # noqa: E402

KS = (1, 3, 5, 10)


# ─── Retrieval settings ──────────────────────────────────────────────────────


class CachedEmbedder(EmbeddingProvider):
    """memblock's default local embedding model, keeping each vector by its text.

    The same messages appear in many questions' histories, so each is embedded
    once. `warm()` does that up front in batches (in several processes when
    `workers` is set); `cache_path` keeps the vectors between runs. The vectors
    are the model's own: a batch and a single call agree to about 1e-7.
    """

    def __init__(self, cache_path: str | None = None) -> None:
        import numpy as np

        from memblock.embeddings import FastEmbedProvider
        from memblock.snapshot import embedding_identity

        self._np = np
        self._inner = FastEmbedProvider()
        self.identity = embedding_identity(self._inner)
        self._path = Path(cache_path) if cache_path else None
        self._vectors: dict[bytes, object] = {}
        self._unsaved = 0
        if self._path and self._path.exists():
            with np.load(self._path) as saved:
                if str(saved["identity"]) == self.identity:
                    self._vectors = {key.tobytes(): row for key, row in zip(saved["keys"], saved["vectors"])}

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @staticmethod
    def _key(text: str) -> bytes:
        return hashlib.sha1(text.encode("utf-8")).digest()

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.warm(texts)
        return [self._vectors[self._key(text)].tolist() for text in texts]

    def warm(self, texts, workers: int | None = None) -> int:
        """Embed every text not seen yet. Returns how many were new."""
        todo: dict[bytes, str] = {}
        for text in texts:
            key = self._key(text)
            if key not in self._vectors and key not in todo:
                todo[key] = text
        if not todo:
            return 0
        ordered = sorted(todo.items(), key=lambda item: len(item[1]))  # similar lengths share a batch
        vectors = self._inner._model.embed([text for _, text in ordered], batch_size=16, parallel=workers)
        for (key, _), vector in zip(ordered, vectors):
            self._vectors[key] = self._np.asarray(vector, dtype=self._np.float32)
        self._unsaved += len(todo)
        return len(todo)

    def save(self) -> None:
        if not self._path or not self._unsaved:
            return
        np = self._np
        keys = np.frombuffer(b"".join(self._vectors), dtype=np.uint8).reshape(-1, 20)
        vectors = np.stack(list(self._vectors.values()))
        temp = self._path.with_name(self._path.name + ".tmp.npz")
        np.savez(temp, keys=keys, vectors=vectors, identity=np.array(self.identity))
        os.replace(temp, self._path)
        self._unsaved = 0


class Stores:
    """Opens the stores a run uses: keyword search only (the default), or with
    the embedding model and the heuristic reranker switched on."""

    def __init__(
        self, embedder: CachedEmbedder | None = None, rerank: bool = False, question_clock: bool = False,
    ) -> None:
        self.embedder = embedder
        self.rerank = rerank
        self.question_clock = question_clock

    def set_clock(self, entry: dict) -> None:
        """With `question_clock`, make memblock read time phrases in this
        question as of the date it was asked. (memblock has no setting for
        this, so the benchmark replaces its clock function.)"""
        if not self.question_clock:
            return
        import memblock.temporal as temporal
        asked = parse_date(entry.get("question_date", ""))
        temporal._now = (lambda: asked) if asked else (lambda: datetime.now(timezone.utc))

    def open(self) -> MemBlock:
        reranker = None
        if self.rerank:
            from memblock.rerankers import HeuristicReranker
            reranker = HeuristicReranker()
        return MemBlock(storage="sqlite:///:memory:", embeddings=self.embedder or False, reranker=reranker)

    @property
    def retrieval(self) -> str:
        name = "keyword + embeddings" if self.embedder else "keyword"
        return name + (" + heuristic reranker" if self.rerank else "")

    def describe(self) -> dict:
        return {
            "retrieval": self.retrieval,
            "embedding_model": self.embedder.identity if self.embedder else None,
            "time_phrases_read_as_of": "the question's date" if self.question_clock else "today",
        }


def texts_in(data: list[dict]) -> list[str]:
    """Every text a run over these questions will embed: each message, each
    session's joined user turns (the baseline's blocks), and each question."""
    texts = []
    for entry in data:
        texts.append(entry["question"])
        for recorded in entry["haystack_sessions"]:
            texts.append("\n".join(m["content"] for m in recorded if m["role"] == "user"))
            texts.extend(m["content"] for m in recorded)
    return [text for text in texts if text.strip()]


# ─── Helpers ─────────────────────────────────────────────────────────────────


def parse_date(text: str) -> datetime | None:
    match = re.match(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})", text or "")
    if not match:
        return None
    year, month, day, hour, minute = map(int, match.groups())
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def turns_of(session: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Split one recorded session into (messages before the first user message,
    [(user message, assistant reply or "")])."""
    leading, turns = [], []
    index = 0
    while index < len(session) and session[index]["role"] != "user":
        leading.append(session[index])
        index += 1
    while index < len(session):
        user = session[index]["content"]
        index += 1
        replies = []
        while index < len(session) and session[index]["role"] != "user":
            replies.append(session[index]["content"])
            index += 1
        turns.append((user, "\n".join(r for r in replies if r)))
    return leading, turns


class Replay:
    """A 'model' that answers with the reply recorded in the dataset."""

    def __init__(self) -> None:
        self.next_reply = ""

    def __call__(self, messages):
        return self.next_reply


def replay_history(mem: MemBlock, entry: dict) -> int:
    """Send one question's whole history through MemorySession. Returns the
    number of messages committed."""
    model = Replay()
    dates = entry.get("haystack_dates") or [""] * len(entry["haystack_sessions"])
    committed = 0
    for recorded, session_id, date in zip(entry["haystack_sessions"], entry["haystack_session_ids"], dates):
        when = parse_date(date)
        leading, turns = turns_of(recorded)
        for message in leading:  # an assistant opener with no user message before it
            if message["content"].strip():
                mem.store_episode(message["content"], role=message["role"], session_id=session_id, happened_at=when)
                committed += 1
        session = MemorySession(
            mem, model, session_id=session_id, memory_read=False,
            clock=(lambda when=when: when) if when else None,
        )
        session._next_turn = 1  # fresh store: skip the look-up of earlier turns
        for user, reply in turns:
            if not user.strip():
                continue
            model.next_reply = reply
            session.chat(user, enable_memory=True)
            committed += 2
    return committed


def rank_sessions(blocks) -> list[str]:
    ranked, seen = [], set()
    for block in blocks:
        sid = block.metadata.session_id
        if sid and sid not in seen:
            seen.add(sid)
            ranked.append(sid)
    return ranked


def score(ranked: list[str], correct: set[str]) -> dict[str, float]:
    out = {}
    for k in KS:
        top = ranked[:k]
        out[f"recall_any@{k}"] = float(any(s in correct for s in top))
        out[f"recall_all@{k}"] = float(all(s in top for s in correct))
    relevances = [1.0 if s in correct else 0.0 for s in ranked[:10]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(correct), 10)))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    return out


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def direct_baseline(entry: dict, stores: Stores) -> list[str]:
    """The existing benchmark's method: one block per session (its user turns
    joined), queried directly."""
    mem = stores.open()
    for recorded, session_id in zip(entry["haystack_sessions"], entry["haystack_session_ids"]):
        doc = "\n".join(m["content"] for m in recorded if m["role"] == "user")
        if doc.strip():
            mem.store(doc, type=BlockType.FACT, source=SourceType.EXPLICIT, session_id=session_id)
    ranked = rank_sessions(mem.query(text_search=entry["question"], limit=50))
    mem.close()
    return ranked


# ─── lme ─────────────────────────────────────────────────────────────────────


def run_lme(data: list[dict], out_path: str | None, stores: Stores | None = None) -> dict:
    stores = stores or Stores()
    rows = []
    started = time.time()
    for index, entry in enumerate(data):
        correct = set(entry["answer_session_ids"])
        stores.set_clock(entry)
        base = score(direct_baseline(entry, stores), correct)

        mem = stores.open()
        messages = replay_history(mem, entry)
        stored = mem.stats()["blocks_by_type"].get("episode", 0)
        ranked = rank_sessions(mem.query(text_search=entry["question"], limit=50))
        through = score(ranked, correct)

        # What the model would actually be shown, with default settings.
        reader = MemorySession(mem, lambda m: "", session_id="question")
        text, ids = reader.recall(entry["question"])
        shown = {mem._storage.get_block(i).metadata.session_id for i in ids}
        in_prompt = float(bool(shown & correct))
        rows.append({
            "question_id": entry["question_id"],
            "question_type": entry["question_type"],
            "messages": messages,
            "episodes_stored": stored,
            "baseline": base,
            "session": through,
            "evidence_in_prompt": in_prompt,
            "memory_tokens": len(text) // 4,
            "history_tokens": sum(len(m["content"]) for s in entry["haystack_sessions"] for m in s) // 4,
            "verified": mem.verify().valid,
        })
        mem.close()
        if (index + 1) % 25 == 0 or index == 0:
            rate = (time.time() - started) / (index + 1)
            print(f"  [{index + 1}/{len(data)}] {rate:.1f}s per question", flush=True)

    summary = {
        **stores.describe(),
        "questions": len(rows),
        "messages_replayed": sum(r["messages"] for r in rows),
        "episodes_stored": sum(r["episodes_stored"] for r in rows),
        "all_stores_verified": all(r["verified"] for r in rows),
        "seconds": round(time.time() - started, 1),
        "baseline": {k: round(mean(r["baseline"][k] for r in rows), 4) for k in rows[0]["baseline"]},
        "session": {k: round(mean(r["session"][k] for r in rows), 4) for k in rows[0]["session"]},
        "evidence_in_prompt": round(mean(r["evidence_in_prompt"] for r in rows), 4),
        "mean_memory_tokens_in_prompt": round(mean(r["memory_tokens"] for r in rows)),
        "mean_history_tokens": round(mean(r["history_tokens"] for r in rows)),
        "by_type": {},
    }
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    for qtype, group in sorted(by_type.items()):
        summary["by_type"][qtype] = {
            "n": len(group),
            "baseline_recall_any@5": round(mean(r["baseline"]["recall_any@5"] for r in group), 4),
            "session_recall_any@5": round(mean(r["session"]["recall_any@5"] for r in group), 4),
            "evidence_in_prompt": round(mean(r["evidence_in_prompt"] for r in group), 4),
        }
    if out_path:
        Path(out_path).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return summary


# ─── noise ───────────────────────────────────────────────────────────────────


def run_noise(
    data: list[dict], pool: list[dict], out_path: str | None, levels=(0, 1, 3), stores: Stores | None = None,
) -> dict:
    """Recall after adding unrelated conversations: 0x, 1x and 3x the size of
    the question's own history, taken from other questions."""
    stores = stores or Stores()
    rng = random.Random(7)
    results = {level: [] for level in levels}
    prompt_hits = {level: [] for level in levels}
    started = time.time()
    for index, entry in enumerate(data):
        correct = set(entry["answer_session_ids"])
        own = set(entry["haystack_session_ids"])
        stores.set_clock(entry)
        mem = stores.open()
        replay_history(mem, entry)
        size = len(entry["haystack_sessions"])
        added = 0
        for level in levels:
            want = size * level
            while added < want:
                other = pool[rng.randrange(len(pool))]
                if other is entry:
                    continue
                pick = rng.randrange(len(other["haystack_sessions"]))
                session_id = f"noise{added}_{other['haystack_session_ids'][pick]}"
                if other["haystack_session_ids"][pick] in own:
                    continue
                # The added conversation keeps its own recorded date. Left undated it
                # would be stamped with today's date, and a question that says "last
                # month" would then match only the added conversations.
                when = parse_date((other.get("haystack_dates") or [""] * (pick + 1))[pick])
                with mem.transaction():
                    for message in other["haystack_sessions"][pick]:
                        if message["content"].strip():
                            mem.store_episode(
                                message["content"], role=message["role"], session_id=session_id, happened_at=when,
                            )
                added += 1
            ranked = rank_sessions(mem.query(text_search=entry["question"], limit=50))
            results[level].append(score(ranked, correct))
            _, ids = MemorySession(mem, lambda m: "", session_id="question").recall(entry["question"])
            shown = {mem._storage.get_block(i).metadata.session_id for i in ids}
            prompt_hits[level].append(float(bool(shown & correct)))
        mem.close()
        if (index + 1) % 10 == 0:
            print(f"  [{index + 1}/{len(data)}] {(time.time() - started) / (index + 1):.1f}s per question", flush=True)
    types = defaultdict(int)
    for entry in data:
        types[entry["question_type"]] += 1
    summary = {
        **stores.describe(),
        "questions": len(data),
        "question_types": dict(sorted(types.items())),
        "levels": {
            f"+{level}x": {
                "recall_any@5": round(mean(r["recall_any@5"] for r in results[level]), 4),
                "recall_any@10": round(mean(r["recall_any@10"] for r in results[level]), 4),
                "ndcg@10": round(mean(r["ndcg@10"] for r in results[level]), 4),
                "evidence_in_prompt": round(mean(prompt_hits[level]), 4),
            }
            for level in levels
        },
    }
    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=1))
    return summary


# ─── growth ──────────────────────────────────────────────────────────────────


def run_growth(data: list[dict], out_path: str | None, question: int = 0) -> dict:
    """One long conversation (a whole question's history as a single session):
    prompt size at each turn, session layer against the full transcript."""
    entry = data[question]
    model = Replay()
    mem = MemBlock(storage="sqlite:///:memory:")
    session = MemorySession(mem, model, session_id="long")
    transcript_chars = 0
    points = []
    turn = 0
    for recorded in entry["haystack_sessions"]:
        _, turns = turns_of(recorded)
        for user, reply in turns:
            if not user.strip():
                continue
            turn += 1
            model.next_reply = reply
            session.chat(user, enable_memory=True)
            full = (transcript_chars + len(user)) // 4  # everything so far + this message
            points.append({"turn": turn, "full_transcript_tokens": full, "session_tokens": session.last_prompt["tokens"]})
            transcript_chars += len(user) + len(reply)
    text, ids = session.recall(entry["question"])
    shown = {mem._storage.get_block(i).metadata.session_id for i in ids}
    marks = [1, 10, 25, 50, 100, 150, 200, len(points)]
    summary = {
        "turns": len(points),
        "at_turn": {str(p["turn"]): {"full_transcript": p["full_transcript_tokens"], "session_layer": p["session_tokens"]}
                    for p in points if p["turn"] in marks},
        "session_layer_max_tokens": max(p["session_tokens"] for p in points),
        "session_layer_mean_tokens": round(mean(p["session_tokens"] for p in points)),
        "full_transcript_final_tokens": points[-1]["full_transcript_tokens"],
        "episodes_stored": mem.stats()["blocks_by_type"].get("episode", 0),
        "final_question": entry["question"],
        "final_question_memory_found": bool(ids),
        "uncommitted_turns": session.status()["uncommitted_turns"],
    }
    mem.close()
    if out_path:
        Path(out_path).write_text(json.dumps({"summary": summary, "points": points}, indent=1))
    return summary


# ─── crash ───────────────────────────────────────────────────────────────────


class Crash(BaseException):
    """The process dying: a BaseException skips every cleanup path."""


def run_crash(trials: int, out_path: str | None) -> dict:
    rng = random.Random(11)
    outcomes = defaultdict(int)
    violations = []
    workdir = tempfile.mkdtemp(prefix="memblock-crash-")

    @contextmanager
    def no_grouping():
        yield

    for trial in range(trials):
        db = f"sqlite:///{os.path.join(workdir, f't{trial}.db')}"
        atomic = trial % 2 == 0  # alternate: SQLite's real atomic(), and an adapter without it
        turns = rng.randint(1, 3)
        messages = turns * 2
        points = ["none", "before_commit_point", "after_commit_point", "before_journal_drop"]
        points += [f"apply:{i}" for i in range(messages)]
        point = rng.choice(points)

        mem = MemBlock(storage=db)
        if not atomic:
            mem._storage.atomic = no_grouping
        mem.store("already here before the crash")
        session = MemorySession(mem, lambda m: "the reply", session_id="s")
        for i in range(turns):
            session.chat(f"turn {i} of trial {trial}")

        def hook(reached, point=point):
            if reached == point:
                raise Crash(reached)

        mem._txn_fault_hook = hook
        crashed = False
        try:
            session.commit_history()
        except Crash:
            crashed = True
        receipts = [r for t in [*session._recent, *session._turns] for r in t.receipts()]
        mem._storage.close()  # the process is gone

        fresh = MemBlock(storage=db)
        if not atomic:
            fresh._storage.atomic = no_grouping
        fresh.recover_transactions(force=True)

        def saved():
            return [b for b in fresh._storage.query_blocks({"type": BlockType.EPISODE})]

        count = len(saved())
        leftovers = [b for b in fresh._storage.get_all_blocks(include_deleted=True) if is_internal(b) or is_staged(b)]
        problems = []
        if count not in (0, messages):
            problems.append(f"partial save: {count} of {messages}")
        if leftovers:
            problems.append(f"{len(leftovers)} hidden rows left")
        if not fresh.verify().valid:
            problems.append("hash chain broken")
        expected_all = point in ("none", "after_commit_point", "before_journal_drop") or point.startswith("apply:")
        if (count == messages) != expected_all:
            problems.append(f"expected {'all' if expected_all else 'none'} saved, got {count}")

        # The app retries the same turns under the same receipts.
        for index, receipt in enumerate(receipts):
            fresh.store_episode(f"retry {index}", role="user", session_id="s", receipt=receipt)
        after_retry = len(saved())
        if after_retry != messages:
            problems.append(f"after retry: {after_retry} of {messages}")
        if len(fresh.query(limit=5, type=BlockType.FACT)) != 1:
            problems.append("earlier data damaged")
        fresh.close()

        outcomes[("crash" if crashed else "clean") + (":saved" if count == messages else ":not saved")] += 1
        if problems:
            violations.append({"trial": trial, "point": point, "atomic": atomic, "problems": problems})

    summary = {
        "trials": trials,
        "violations": len(violations),
        "outcomes": dict(sorted(outcomes.items())),
        "details": violations[:10],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=1))
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("lme", "noise", "growth"):
        p = sub.add_parser(name)
        p.add_argument("data_file")
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--out", default=None)
        if name != "growth":
            p.add_argument("--embeddings", action="store_true", help="keyword + vector search (local model)")
            p.add_argument("--rerank", action="store_true", help="add the heuristic reranker")
            p.add_argument("--question-clock", action="store_true",
                           help="read time phrases in a question as of the question's date")
            p.add_argument("--embedding-cache", default=None, help="an .npz file that keeps vectors between runs")
            p.add_argument("--workers", type=int, default=None, help="processes used to embed the texts up front")
    p = sub.add_parser("crash")
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.command == "crash":
        summary = run_crash(args.trials, args.out)
    else:
        with open(args.data_file) as handle:
            data = json.load(handle)
        if args.command == "growth":
            summary = run_growth(data, args.out)
        else:
            if args.command == "lme":
                sample = data[: args.limit] if args.limit else data
                to_embed = sample
            else:
                # The file is ordered by question type, so take an even sample of it
                # rather than its first N; noise is still drawn from all of it.
                sample = random.Random(3).sample(data, args.limit) if args.limit else data
                to_embed = data
            embedder = None
            if args.embeddings:
                embedder = CachedEmbedder(args.embedding_cache)
                started = time.time()
                new = embedder.warm(texts_in(to_embed), workers=args.workers)
                embedder.save()
                print(f"  embedded {new} new texts in {time.time() - started:.0f}s", flush=True)
            stores = Stores(embedder, rerank=args.rerank, question_clock=args.question_clock)
            if args.command == "lme":
                summary = run_lme(sample, args.out, stores)
            else:
                summary = run_noise(sample, data, args.out, stores=stores)
            if embedder:
                embedder.save()  # texts met on the way (joined replies, for one)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
