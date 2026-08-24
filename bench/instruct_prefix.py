#!/usr/bin/env python3
"""
Does Qwen3-Embedding's instruction prefix help on THIS store?

WHY THIS CAN BE ANSWERED FOR FREE

Qwen3-Embedding is instruction-aware, and the format is asymmetric:
the prefix goes on the QUERY, the document is embedded raw. Which
means the 224 vectors already in the store are exactly what a
prefixed run needs to search against -- they were stored raw and
they stay raw. Nothing has to be re-embedded, nothing is written, and
the experiment costs one extra embedding call per question.

That asymmetry is the same rule Forge already decided on 2026-08-22
for query expansion -- extend the QUERY, never the FACT -- arrived at
independently from the model's own spec. Worth noticing, since it
means the prefix is not a competing idea to the expansion lot but its
first and cheapest component.

WHAT IT DOES NOT ANSWER

Whether the prefix helps at INDEX time. Qwen's retrieval recipe puts
no instruction on documents, so there is nothing to test there, but a
store whose documents are themselves questions -- which is what one
entry per exchange produces -- is not the case the recipe was written
for. If the numbers below are ambiguous, that is the next thing to
suspect, and it is NOT free: it needs every vector rewritten.

READING IT

The gap is the number, same as bench/recall_distance.py: worst hit
versus best miss. A prefix that improves every distance by the same
amount has done nothing -- distances are only meaningful against each
other. What would count as a win is the gap widening, or a hit that
was outranked by noise coming back to rank 1.

A hit is scored on the entry --expect NAMES, not on whatever came back
first, and those are different numbers whenever rank is not 1. This is
the one place this harness deliberately reads differently from
recall_distance.py, which keeps the closest row and warns: that file
is calibrating a cutoff, and a cutoff acts on whatever the store
returns. This one is asking whether the instruction moved THE RIGHT
ENTRY, and comparing the distance to an unrelated row in one column
against the distance to the answer in the other measures nothing at
all. The closest row is still printed in brackets, because an archived
refusal outranking the answer is exactly the failure this store has.

Without --expect there is nothing to name, so the hit column falls
back to the closest row and the verdict says so. A question you cannot
score is a question this harness should not be quietly averaging in.

    bench/in_container.sh instruct_prefix --db /tmp/real_copy.db \\
        --hit "Quel processeur a mon NiPoGi ?" --expect 308 \\
        --miss "Comment s'appelle mon chat ?"

The six-command copy dance lives in bench/in_container.sh now -- the
faults in it are silent (a merged /tmp/arm holding two checkouts, a
harness pointed at the real store) and it was duplicated across every
file here.

Read-only. It never writes to the database it is given, but pass a
copy anyway -- the habit is what keeps a benchmark out of production.
"""

from __future__ import annotations

import argparse
import os
import sys

from _harness import placeholders, read_row, split_misplaced

# The task description Qwen's format expects. Theirs is written for
# web search; this one says what Forge actually stores, because the
# instruction is embedded along with the question and a description of
# the wrong corpus is a description of the wrong corpus.
DEFAULT_INSTRUCT = (
    "Given a question asked in conversation, retrieve the stored facts, "
    "decisions and past exchanges that answer it"
)


class _instruct:
    """
    Force what rag.search prefixes with, for the duration of one call.

    Necessary from the moment the prefix shipped: rag.search reads
    EMBEDDING_QUERY_INSTRUCT itself, so once it is set in production
    this harness's "RAW" column was silently already prefixed and its
    "PREFIXED" column was prefixed TWICE. The 2026-08-23 run after
    deployment reported +0.0008 of change and recommended keeping
    queries raw -- a verdict about stacking two instructions, printed
    as if it were about using one.

    Setting it here rather than asking the operator to export an
    environment variable is the point: a measurement that depends on
    the deployed configuration measures the deployment, not the thing.
    """

    def __init__(self, rag_module, value: str):
        self.rag, self.value = rag_module, value

    def __enter__(self):
        self.previous = self.rag.EMBEDDING_QUERY_INSTRUCT
        self.rag.EMBEDDING_QUERY_INSTRUCT = self.value
        return self

    def __exit__(self, *exc):
        self.rag.EMBEDDING_QUERY_INSTRUCT = self.previous
        return False


def _cell(row: tuple[float | None, int | None, float | None]) -> str:
    """One column of one line: what the named entry scored, and what won."""
    scored, rank, closest = row
    if scored is None:
        return f"{'--':<24}"
    rank_text = "-" if rank is None else str(rank)
    return f"{scored:.4f} rank={rank_text:<3} [{closest:.4f}]"


def _scoreable(
    expect: str | None, row: tuple[float | None, int | None, float | None]
) -> bool:
    """
    Whether this question's distance may go into the gap.

    An expectation that came back at any rank is scored on that row.
    An expectation that did not come back at all is not scored at all:
    read_row falls back to the closest distance so the line still
    prints, and averaging that into the verdict is precisely the fault
    that made the sibling harness announce a gap "too tight to act on"
    about a threshold that was fine.
    """
    scored, rank, _ = row
    if not isinstance(scored, float):
        return False
    return rank is not None or expect is None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--hit", action="append", default=[], metavar="QUESTION")
    parser.add_argument("--expect", action="append", default=[], metavar="ID")
    parser.add_argument("--miss", action="append", default=[], metavar="QUESTION")
    args = parser.parse_args()

    bad = placeholders(args.hit + args.miss)
    if bad:
        print("these look like unfilled placeholders, not questions:")
        for q in bad:
            print(f"  {q}")
        return 1
    if not args.hit or not args.miss:
        print("need at least one --hit and one --miss")
        return 1
    if args.expect and len(args.expect) != len(args.hit):
        print(
            f"--expect given {len(args.expect)} times for {len(args.hit)} --hit "
            "questions. Matched by position: one each, or none at all."
        )
        return 1
    if not os.path.exists(args.db):
        print(f"{args.db} does not exist")
        return 1

    os.environ["RAG_DB_FILE"] = args.db
    from forge import rag

    expects = args.expect or [None] * len(args.hit)
    conn = rag.get_connection()
    raw_hits, pre_hits, raw_misses, pre_misses = [], [], [], []
    raw_rows, pre_rows = [], []

    try:
        print(f"instruction: {args.instruct!r}\n")
        print(f"{'':<44} {'RAW':<24} {'PREFIXED':<24}")
        print("=" * 94)

        print("HITS  (distance to the --expect entry, [closest row] beside it)")
        for question, expect in zip(args.hit, expects):
            with _instruct(rag, ""):
                raw = read_row(
                    rag.search(conn, query=question, top_k=args.top_k), expect
                )
            with _instruct(rag, args.instruct):
                pre = read_row(
                    rag.search(conn, query=question, top_k=args.top_k), expect
                )
            raw_rows.append((question, expect, raw[1]))
            pre_rows.append((question, expect, pre[1]))
            # Both columns or neither. A question scored in one and
            # dropped from the other would make the gap a comparison
            # between two different sets of questions, which is the
            # same class of mistake as comparing two different rows.
            if _scoreable(expect, raw) and _scoreable(expect, pre):
                raw_hits.append(raw[0])
                pre_hits.append(pre[0])
            print(f"  {question[:42]:<42} {_cell(raw)} {_cell(pre)}")
            if expect is not None and raw[1] != pre[1]:
                print(f"       #{expect} moved: rank {raw[1]} -> {pre[1]}")

        print("\nMISSES  (further is better here)")
        for question in args.miss:
            with _instruct(rag, ""):
                raw = read_row(rag.search(conn, query=question, top_k=args.top_k), None)
            with _instruct(rag, args.instruct):
                pre = read_row(rag.search(conn, query=question, top_k=args.top_k), None)
            if isinstance(raw[0], float):
                raw_misses.append(raw[0])
            if isinstance(pre[0], float):
                pre_misses.append(pre[0])
            print(f"  {question[:42]:<42} {_cell(raw)} {_cell(pre)}")

        # Two states under one warning is how a reader draws the
        # wrong conclusion from a correct message. Second place is
        # scored and counted; absent is not scoreable at all. The
        # first version printed both under one heading whose
        # explanation only covered the second, so a question that WAS
        # in the verdict read as though it had been thrown out.
        outranked, absent = split_misplaced(raw_rows + pre_rows)
        if outranked:
            print("\n  /!\\ named entry came back, but not first, for:")
            for question in outranked:
                print(f"        {question}")
            print("      Still scored on the right row, so these ARE in the")
            print("      verdict. Something else in the store is closer to the")
            print("      question than the entry that answers it -- the bracket")
            print("      shows what.")
        if absent:
            print("\n  /!\\ named entry did not come back AT ALL for:")
            for question in absent:
                print(f"        {question}")
            print("      Left out of the verdict: there is no distance to the")
            print("      right row to compare. Fix retrieval before reading this.")
        if not args.expect:
            print("\n  No --expect given, so the hit column is whatever came back")
            print("  first, which is the right entry only when it is. This store")
            print("  holds archived exchanges that contain the question, and they")
            print("  outrank the answer; name the entries.")
    finally:
        conn.close()

    if not (raw_misses and pre_misses):
        print("\nno distances came back -- is the embedding server up?")
        return 1
    if not (raw_hits and pre_hits):
        # Distances came back; none of them were about the entry that
        # was supposed to answer. Printing a gap here would be a
        # number about the wrong rows, which is the fault this harness
        # was carrying.
        print("\nno hit could be scored: every --expect entry was missing from")
        print("its results. There is no gap to report -- this is a retrieval")
        print("failure, and a threshold is not what fixes it.")
        return 1

    gap_raw = min(raw_misses) - max(raw_hits)
    gap_pre = min(pre_misses) - max(pre_hits)

    print("\n=== VERDICT ===")
    if len(raw_hits) < 3 or len(raw_misses) < 3:
        # recall_distance refuses outright below three of each, and it
        # is right to: it hands back a threshold, and a threshold from
        # two points is a coin flip with a decimal place. This one
        # reports a DIRECTION, which survives a thin sample better --
        # the 2026-08-23 result was trustworthy because all six
        # questions moved the way one mechanism predicts, not because
        # of the size of the gap. So: said out loud, not refused.
        print(
            f"  (on {len(raw_hits)} scored hit(s) and {len(raw_misses)} miss(es) "
            "-- read the direction, not the number)"
        )
    print(
        f"  raw       worst hit {max(raw_hits):.4f} | best miss "
        f"{min(raw_misses):.4f} | gap {gap_raw:+.4f}"
    )
    print(
        f"  prefixed  worst hit {max(pre_hits):.4f} | best miss "
        f"{min(pre_misses):.4f} | gap {gap_pre:+.4f}"
    )
    print()

    # A prefix that shifts everything equally has changed nothing. Only
    # the gap is read, and only a change large enough not to be noise
    # on a handful of questions is called a result.
    delta = gap_pre - gap_raw
    if abs(delta) < 0.02:
        print(f"  NO REAL CHANGE ({delta:+.4f}). The prefix moved the absolute")
        print("  distances but not their separation, which is the only thing")
        print("  a threshold can act on. Keep the queries raw -- an extra")
        print("  wrapper on every search should have to earn its place.")
    elif delta > 0:
        print(f"  BETTER by {delta:+.4f} of gap. Worth wiring into the recall")
        print("  path -- on the query only, never on what gets stored.")
    else:
        print(f"  WORSE by {delta:+.4f} of gap. The instruction is describing a")
        print("  corpus this store is not; try --instruct with another wording")
        print("  before concluding the mechanism does not apply.")

    if gap_pre <= 0 and gap_raw <= 0:
        print("\n  Both are negative: a miss is still closer than the worst")
        print("  hit either way. The prefix is not the thing standing between")
        print("  this store and a usable threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
