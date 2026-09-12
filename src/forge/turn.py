"""
The current turn's raw user input, and what its shape says.

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

#: Past this, a trailing "?" is part of a longer message rather than
#: the whole of one -- "faut-il le faire dans src/forge ou dans tests/ ?"
#: is an answer that happens to end in a question mark.
_MAX_QUESTION_CHARS = 40


def is_question(text: str) -> bool:
    """
    Whether this message is the user asking rather than telling.

    ONE definition, and it had two before this. delegation.py held it
    twice -- `_looks_like_a_question` and `_is_question`, plus two
    module-level constants of the same name where the second silently
    overwrote the first -- and forge/tools/memory.py was about to make
    it three. Two copies of a decision that has to stay identical is
    how a fix reaches one of them, which is the reason non_answer.py
    and lang.py exist.

    What it decides is small and it is decidable, which is the whole
    argument for deciding it here rather than asking a model. The two
    callers use it for opposite purposes and that is fine: delegation
    refuses to record a question as the answer to its own prompt, and
    the memory tool refuses to write a question into the store as a
    fact about the user.

    Bounded by length because a long message ending in "?" is a
    message with a question mark at the end, not a question. Found on
    delegation's first real run: "C'est à dire ?" had been recorded as
    the workspace of a job.
    """
    stripped = text.strip()
    return stripped.endswith("?") and len(stripped) <= _MAX_QUESTION_CHARS


def set_input(text: str) -> None:
    _local.user_input = text


def get_input() -> str:
    """The raw message, or "" outside a run (direct calls, tests)."""
    return getattr(_local, "user_input", "")


def clear() -> None:
    _local.user_input = ""
