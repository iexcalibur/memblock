"""Background extraction worker for non-blocking memory extraction.

Runs LLM extraction in a background thread pool so store() returns instantly.
Extracted FACT/ENTITY/RELATION blocks appear asynchronously and are linked
to the original block via DERIVED_FROM.

Usage:
    mem = MemBlock(
        storage="sqlite:///./memory.db",
        extract_provider="gemini",
        extract_api_key="...",
        auto_extract_on_store=True,
        background_extract=True,  # ← enables background mode
    )

    # store() returns immediately — extraction happens in background
    block = mem.store("I just adopted a golden retriever named Max")

    # Later, check status or wait for all pending extractions
    mem.extraction_pending     # → number of pending tasks
    mem.wait_for_extractions() # → blocks until all done
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ExtractionTask:
    """Tracks a single background extraction job."""
    block_id: str
    content: str
    future: Future | None = None


class BackgroundExtractor:
    """
    Thread-pool based background extraction worker.

    Accepts extraction jobs and runs them in a bounded thread pool.
    Each job extracts structured blocks from content and links them
    to the source block via DERIVED_FROM.

    Thread-safe: multiple store() calls can submit jobs concurrently.
    """

    def __init__(
        self,
        max_workers: int = 2,
        on_complete: Callable[[str, list[str]], None] | None = None,
        on_error: Callable[[str, Exception], None] | None = None,
    ) -> None:
        """
        Args:
            max_workers: Max concurrent extraction threads (default 2).
                Keep low to avoid LLM rate limits.
            on_complete: Callback(block_id, extracted_ids) on success.
            on_error: Callback(block_id, exception) on failure.
        """
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="memblock-extract",
        )
        self._on_complete = on_complete
        self._on_error = on_error
        self._pending: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._shutdown = False

        # Stats
        self._completed_count = 0
        self._failed_count = 0
        self._total_submitted = 0

    @property
    def pending_count(self) -> int:
        """Number of extraction jobs still running or queued."""
        with self._lock:
            # Clean up finished futures
            self._cleanup_finished()
            return len(self._pending)

    @property
    def stats(self) -> dict[str, int]:
        """Extraction worker statistics."""
        with self._lock:
            self._cleanup_finished()
            return {
                "pending": len(self._pending),
                "completed": self._completed_count,
                "failed": self._failed_count,
                "total_submitted": self._total_submitted,
            }

    def submit(
        self,
        block_id: str,
        content: str,
        extract_fn: Callable[[str], Any],
        link_fn: Callable[[str, str], None],
    ) -> None:
        """
        Submit an extraction job to run in the background.

        Args:
            block_id: The source block ID to link extracted blocks to.
            content: The text content to extract from.
            extract_fn: Callable that takes content and returns ExtractionResult.
            link_fn: Callable(extracted_id, source_id) to create DERIVED_FROM edge.
        """
        if self._shutdown:
            logger.warning("BackgroundExtractor is shut down, ignoring submit for %s", block_id)
            return

        future = self._pool.submit(
            self._run_extraction,
            block_id,
            content,
            extract_fn,
            link_fn,
        )

        with self._lock:
            self._pending[block_id] = future
            self._total_submitted += 1

    def _run_extraction(
        self,
        block_id: str,
        content: str,
        extract_fn: Callable[[str], Any],
        link_fn: Callable[[str, str], None],
    ) -> list[str]:
        """Execute extraction and linking in a worker thread."""
        try:
            result = extract_fn(content)

            extracted_ids = []
            if result and hasattr(result, 'block_ids') and result.block_ids:
                for extracted_id in result.block_ids:
                    try:
                        link_fn(extracted_id, block_id)
                        extracted_ids.append(extracted_id)
                    except Exception as link_err:
                        logger.debug("Failed to link %s → %s: %s", extracted_id, block_id, link_err)

            with self._lock:
                self._completed_count += 1

            if self._on_complete:
                try:
                    self._on_complete(block_id, extracted_ids)
                except Exception:
                    pass  # callback errors shouldn't break extraction

            logger.debug(
                "Background extraction for %s complete: %d blocks extracted",
                block_id, len(extracted_ids),
            )
            return extracted_ids

        except Exception as e:
            with self._lock:
                self._failed_count += 1

            if self._on_error:
                try:
                    self._on_error(block_id, e)
                except Exception:
                    pass

            logger.warning("Background extraction failed for %s: %s", block_id, e)
            return []

    def wait(self, timeout: float | None = None) -> None:
        """
        Block until all pending extractions complete.

        Args:
            timeout: Max seconds to wait. None = wait forever.
        """
        while True:
            with self._lock:
                futures = list(self._pending.values())
            if not futures:
                break

            for future in futures:
                try:
                    future.result(timeout=timeout)
                except Exception:
                    pass  # errors already handled in _run_extraction

            with self._lock:
                self._cleanup_finished()
                if not self._pending:
                    break

    def shutdown(self, wait: bool = True) -> None:
        """
        Shut down the extraction worker pool.

        Args:
            wait: If True, wait for pending tasks to complete.
        """
        self._shutdown = True
        self._pool.shutdown(wait=wait)
        with self._lock:
            self._pending.clear()

    def _cleanup_finished(self) -> None:
        """Remove completed futures from pending dict. Must hold self._lock."""
        done_ids = [
            bid for bid, f in self._pending.items()
            if f.done()
        ]
        for bid in done_ids:
            del self._pending[bid]
