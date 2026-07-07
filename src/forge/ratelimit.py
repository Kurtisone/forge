"""
In-memory sliding-window rate limiter for the HTTP API.

No external dependency (no redis, no slowapi) -- matches the rest of
Forge's local, single-process posture (see memory.py's plain-JSON
store, trace.py's JSONL file). Counters live in a process-local dict,
so this only limits per-worker: running uvicorn with multiple workers
gives each its own independent counter, effectively multiplying the
limit by worker count. Fine for the single-worker deployment this
project documents (see the Containerfile); worth knowing if that ever
changes.
"""

import threading
import time
from collections import defaultdict, deque

from forge.config import (
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)


def check(key: str) -> tuple[bool, int]:
    """
    Record a hit for `key` and report whether it's within the limit.

    Returns (allowed, retry_after_seconds). retry_after_seconds is 0
    when allowed is True.
    """
    if not RATE_LIMIT_ENABLED:
        return True, 0

    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    with _lock:
        hits = _hits[key]
        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= RATE_LIMIT_REQUESTS:
            retry_after = int(hits[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1
            return False, max(retry_after, 1)

        hits.append(now)
        return True, 0


def reset() -> None:
    """Test helper: clear every counter."""
    with _lock:
        _hits.clear()
