"""
The current turn's raw user input.

Tools receive only the router's `content` -- its restatement of the
request. That is fine for every tool that acts on a payload, and
wrong for delegation, which turns the request into an objective an
implementer will be held to. The first real runs showed why: asked
"délègue un truc", the router once answered with `content` lifted
from an EARLIER message in the history, so the spec's objective came
from a request that had nothing to do with the turn.

Widening the tool contract to pass both would touch all fourteen
tools to serve one. This follows subtrace.py instead: a module the
orchestrator sets at the top of a run and clears when it finishes.

Thread-local because api.py serves chat turns from a
ThreadPoolExecutor with two workers, so two runs can be in flight at
once. A plain module global would let one turn read the other's
input, which for delegation means writing someone else's request
into a job.
"""

import threading

_local = threading.local()


def set_input(text: str) -> None:
    _local.user_input = text


def get_input() -> str:
    """The raw message, or "" outside a run (direct calls, tests)."""
    return getattr(_local, "user_input", "")


def clear() -> None:
    _local.user_input = ""
