"""
The decisions two harnesses make identically, in one place.

bench/recall_distance.py and bench/instruct_prefix.py ask different
questions -- where does a cutoff go, and did the query instruction
help -- but they read a result list the same way: which row was the
one that should have answered, and is the number about to be printed
computed over that row or over something else.

Both had their own copy of that reading. On 2026-08-23 the copy in
recall_distance.py was fixed twice in one afternoon (an --expect that
matched nothing, then a warning whose condition excluded the case it
was written for) and the copy in instruct_prefix.py was not touched,
so the harness that justified turning the query instruction on kept
scoring a failed retrieval as a mediocre hit. Two copies of a
decision that has to stay identical is how a fix reaches one of them,
which is the same reason forge/lang.py exists.

These are pure functions on purpose: no store, no embedding server,
no arguments parsed. tests/test_bench_harness.py exercises them,
which is the coverage bench/ did not have when five faults reached
the Deck in one session.

NOTE for the copy-into-the-container procedure documented at the top
of each harness: this module has to travel with them. An import that
fails is loud, so this is a nuisance rather than a trap, but it is a
second `podman cp` line.
"""

from __future__ import annotations

import re

# Anything shaped like a slot someone forgot to fill. The 2026-08-23
# run sent "<la 3e question hit du 22/08>" and "<la question
# anniversaire cousin du 22/08>" straight to the embedding server;
# they matched "Merci" and "Bonjour" at 0.8962 and 0.9601, and the
# harness printed NO GAP -- DO NOT SET A THRESHOLD off the back of it.
#
# That was the SECOND time a harness produced a confident verdict from
# its own boilerplate. The first was the placeholder sentences in the
# help text, which is why recall_distance's _MIN_QUESTIONS exists. A
# minimum count does not catch this one: three placeholders are still
# three questions. So the shape gets checked too.
PLACEHOLDER = re.compile(r"[<>]|\.\.\.|^\s*$|\bTODO\b|\bXXX\b")


def placeholders(questions: list[str]) -> list[str]:
    """The questions that look like unfilled slots rather than questions."""
    return [q for q in questions if PLACEHOLDER.search(q)]


def find_rank(
    results: list[dict], fixture_text: str | None, expect_id: str | None
) -> int | None:
    """
    Where the entry that should have answered came back, 1-based.

    Two ways to name it, because there are two modes. A planted
    fixture is found by its text; with --no-plant the operator names
    an id, since there is no planted text to look for.

    None means "not in the results at all", which is NOT the same as
    "no expectation given" -- see `misplaced`, which is where that
    distinction has to be made, because this function cannot tell them
    apart and once printed as `rank=None` neither could anyone else.
    """
    if fixture_text:
        return next(
            (
                i + 1
                for i, r in enumerate(results)
                if (r.get("content") or "").startswith(fixture_text[:40])
            ),
            None,
        )
    if expect_id is not None:
        return next(
            (
                i + 1
                for i, r in enumerate(results)
                if str(r.get("id")) == str(expect_id)
            ),
            None,
        )
    return None


def misplaced(rows: list[tuple[str, str | None, int | None]]) -> list[str]:
    """
    The questions whose expected entry did not come back first.

    rows are (question, expect_id, rank). An expectation that was
    never given is not a failure; an expectation that came back second
    is; and an expectation that did not come back AT ALL is the worst
    of the three, which is exactly the case the first version of this
    check let through -- it tested `rank != 1` while excluding None,
    so a question whose answer was nowhere in the results passed
    silently and its distance went into the gap as though it were a
    hit.
    """
    return [q for q, expect, rank in rows if expect is not None and rank != 1]


def read_row(
    results: list[dict], expect_id: str | None
) -> tuple[float | None, int | None, float | None]:
    """
    The three numbers a row of a comparison table needs.

    Returns (scored, rank, closest):

      scored   the distance to the entry the operator NAMED, when they
               named one and it came back. Otherwise the closest
               distance, because there is nothing better to read.
      rank     where the named entry landed, 1-based, or None.
      closest  the first row's distance, always -- "what came back"
               stays visible next to "what should have".

    `scored` and `closest` differing is the whole point. An A/B on the
    query instruction asks whether the RIGHT entry moved closer, and
    the closest row is only the right entry when rank is 1. On
    2026-08-23 "Tu peux me lister mon matériel ?" scored 0.9083
    against an entry about tools while every hardware fact sat outside
    the top 5, and that number was read as a mediocre hit.

    When the named entry did not come back at all, `scored` falls back
    to the closest row and `rank` is None -- the caller has to decide
    what to do with a question it could not score, and `misplaced`
    is what tells it which ones those are.
    """
    if not results:
        return None, None, None
    closest = results[0].get("distance")
    rank = find_rank(results, None, expect_id)
    if rank is None:
        return closest, None, closest
    scored = results[rank - 1].get("distance")
    return scored, rank, closest
