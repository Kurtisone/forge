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

Keys are expired as well as counted (audit M-2). The dict used to be
append-only: one entry per client IP that ever made a request, kept
for the life of the process. That's a slow leak on any instance
reachable by more than one address, and a fast one for anyone who can
vary their source IP -- the rate limiter itself becoming the thing
that exhausts memory is a poor trade for what it protects.
"""

import threading
import time
from collections import deque

from forge.config import (
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)

# Ceiling on tracked keys, enforced independently of the timed sweep
# below. The sweep bounds memory over time; this bounds it during a
# single window, when a flood of distinct source addresses could
# otherwise grow the dict faster than any schedule collects it.
_MAX_TRACKED_KEYS = 10_000

_lock = threading.Lock()
_hits: dict[str, deque] = {}
_last_sweep = 0.0


def _sweep_locked(now: float) -> None:
    """Drop every key whose hits have all aged out. Caller holds _lock.

    A key is only removed when its most recent hit is older than the
    window, i.e. when keeping it and dropping it are observationally
    identical: a client that comes back gets a fresh deque and the
    same allowance it would have had. Expiry here is bookkeeping, not
    policy -- it must never hand anyone a budget they hadn't earned.
    """
    global _last_sweep
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    for key in [k for k, hits in _hits.items() if not hits or hits[-1] < window_start]:
        del _hits[key]
    _last_sweep = now


def _evict_oldest_locked(target: int) -> None:
    """Last resort when a sweep didn't get the dict under the ceiling.

    Only reachable if _MAX_TRACKED_KEYS distinct clients are all
    genuinely active inside one window, which on a personal instance
    means something is wrong rather than busy. Evicting the
    least-recently-seen entries does give those clients a fresh
    allowance -- but they are, by construction, the ones who have
    hammered least recently, and the alternative is the process dying,
    which lifts the limit for everyone at once.
    """
    by_age = sorted(_hits.items(), key=lambda item: item[1][-1] if item[1] else 0.0)
    for key, _ in by_age[: max(len(_hits) - target, 0)]:
        del _hits[key]


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
        # Amortised: one pass over the dict per window, not per
        # request. The ceiling check is what makes a burst inside a
        # single window bounded too.
        if now - _last_sweep >= RATE_LIMIT_WINDOW_SECONDS:
            _sweep_locked(now)
        if len(_hits) > _MAX_TRACKED_KEYS:
            _sweep_locked(now)
            if len(_hits) > _MAX_TRACKED_KEYS:
                _evict_oldest_locked(_MAX_TRACKED_KEYS)

        hits = _hits.get(key)
        if hits is None:
            # Plain dict, not defaultdict: reading a key must not
            # create one. The sweep above iterates this dict, and a
            # container that grows on read is the kind of thing that
            # makes a leak fix leak.
            hits = _hits[key] = deque()

        while hits and hits[0] < window_start:
            hits.popleft()

        if len(hits) >= RATE_LIMIT_REQUESTS:
            retry_after = int(hits[0] + RATE_LIMIT_WINDOW_SECONDS - now) + 1
            return False, max(retry_after, 1)

        hits.append(now)
        return True, 0


def tracked_keys() -> int:
    """Number of client keys currently held. Exposed for tests and for
    anyone wanting to confirm the dict isn't growing without bound."""
    with _lock:
        return len(_hits)


def reset() -> None:
    """Test helper: clear every counter."""
    global _last_sweep
    with _lock:
        _hits.clear()
        _last_sweep = 0.0
