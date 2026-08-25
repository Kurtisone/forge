#!/usr/bin/env python3
"""
Which channel reaches which entry, and what the word channel drags in
behind it.

    bench/in_container.sh rag_hybrid \\
        --hit "Tu peux me lister mon matériel ?" --expect 307 \\
        --hit "Sur quoi tournent mes conteneurs ?" --expect 313 \\
        --miss "Comment s'appelle mon chat ?"

WHAT THIS ONE ASKS THAT THE OTHERS DO NOT

bench/recall_distance.py places a threshold and bench/recall_expansion.py
scored a rescue pass; both of them measure DISTANCES, and both of them
therefore measure one channel. This harness exists because the failure
it is aimed at produces no distance worth reading: #307 and #313 are
in the store and no phrasing brings them back at all. There is no
number to place. The question is binary -- reached, or not -- and it
has to be asked once per channel.

So there is no gap block here and no suggested cutoff. Suggesting one
would be inventing the very thing the lexical channel was designed
not to need (see RECALL_LEXICAL_MAX_DF in config.py: its admission
rule lives on the query side, computed from the store, precisely so
that no second number needs calibrating and tagging).

WHAT IT COUNTS, BOTH SIDES

  REACHED     a --hit whose --expect entry the word channel returns
              and the vector channel does not deliver (absent from its
              top-k, or inside it but beyond the cutoff). This is the
              win, and it is the only reason to turn RECALL_LEXICAL on.

  INTRUDERS   a --miss that the word channel answers. A question that
              was correctly refused now returns rows, and synthesis
              will write a sentence out of them. This is the cost, it
              is the same cost the expansion pass was measured on and
              turned off for, and a run that only counts the wins is
              not a measurement.

  WRONG ENTRY a --hit where the word channel puts something ahead of
              the entry that answers -- reached, and answered out of
              the wrong row.

Both sides come from the SAME rows, so read the candidate lists. A row
that shares a rare word with a question without answering it is what
this store is full of: the archived refusals (#35/#36/#37) quote the
question they failed to answer, which makes them excellent lexical
matches for it.

--max-df MAY BE GIVEN SEVERAL TIMES and prints one column each. It is
the one number this channel has, it is a starting value rather than a
measurement, and sweeping it is how it stops being one. Lower admits
fewer words (fewer intruders, fewer rescues); higher admits more.

--no-vector skips every embedding call, so this runs with the
embedding server down or asleep. The word channel needs no model at
all -- which is also why it costs nothing in a deployment.

The database is a COPY, always: bench/in_container.sh makes one and
points --db at it. Nothing here writes, but the habit is what keeps a
harness out of production.
"""

from __future__ import annotations

import argparse
import os
import sys

from _harness import find_rank, placeholders


def _row(results: list[dict], expect: str | None) -> tuple[int | None, dict | None]:
    """Where the named entry landed and the row itself, or (None, None)."""
    rank = find_rank(results, None, expect)
    return (rank, results[rank - 1]) if rank else (None, None)


def _vector_cell(results: list[dict], expect: str | None, cutoff: float | None) -> str:
    rank, row = _row(results, expect)
    if row is None:
        if not results:
            return "nothing"
        closest = results[0].get("distance")
        shown = f"{closest:.4f}" if isinstance(closest, float) else "--"
        return f"absent    [nearest {shown} #{results[0].get('id')}]"
    distance = row.get("distance")
    verdict = ""
    if cutoff is not None and isinstance(distance, float):
        verdict = "  CUT" if distance > cutoff else "  kept"
    return f"r{rank:<2} {distance:.4f}{verdict}"


def _lexical_cell(results: list[dict], expect: str | None) -> str:
    if not results:
        return "nothing"
    if expect is None:
        # No entry was named -- there is nothing to be absent. Saying
        # "absent" here reads as a failure on a --miss, where rows
        # coming back at all is the finding.
        return f"{len(results)} row(s)"
    rank, row = _row(results, expect)
    if row is None:
        return f"absent    [{len(results)} other row(s)]"
    return f"r{rank:<2} bm25 {row.get('score', 0.0):.2f}"


def _delivered(results: list[dict], expect: str | None, cutoff: float | None) -> bool:
    """
    Would the vector channel actually have handed this entry to
    synthesis?

    Not the same question as "did it come back". An entry inside the
    top-k but beyond the cutoff is dropped before synthesis ever sees
    it, and counting that as delivered is how a rescue that fires
    every day gets scored as unnecessary.
    """
    _, row = _row(results, expect)
    if row is None:
        return False
    distance = row.get("distance")
    if cutoff is None or not isinstance(distance, float):
        return True
    return distance <= cutoff


def _print_candidates(results: list[dict], label: str) -> None:
    print(f"       {label}")
    for row in results:
        score = row.get("score")
        distance = row.get("distance")
        marker = (
            f"{distance:.4f}"
            if isinstance(distance, float)
            else (f"{score:7.2f}" if isinstance(score, float) else "   --  ")
        )
        shown = " ".join(str(row.get("content", "")).split())[:70]
        print(
            f"         #{row.get('id'):<5} {marker}  {row.get('kind', ''):<15} {shown}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lexical-top-k", type=int, default=3)
    parser.add_argument(
        "--max-df",
        action="append",
        type=float,
        default=[],
        metavar="FRACTION",
        help=(
            "Share of the store a word may appear in and still be searched "
            "for. Repeatable: one column per value. Defaults to the "
            "container's RECALL_LEXICAL_MAX_DF."
        ),
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help=(
            "RECALL_MAX_DISTANCE, for reading the vector column only. Never "
            "applied to the word channel -- that is the whole point of the "
            "branch. Defaults to the container's."
        ),
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Word channel only: no embedding call, works with the server down.",
    )
    parser.add_argument("--hit", action="append", default=[], metavar="QUESTION")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="ID",
        help="Matched by position to --hit. Use - for one you cannot name yet.",
    )
    parser.add_argument("--miss", action="append", default=[], metavar="QUESTION")
    args = parser.parse_args()

    bad = placeholders(args.hit + args.miss)
    if bad:
        print("these look like unfilled placeholders, not questions:")
        for question in bad:
            print(f"  {question}")
        return 1
    if not args.hit and not args.miss:
        print("need at least one --hit or one --miss")
        return 1
    if args.expect and len(args.expect) != len(args.hit):
        print(
            f"--expect given {len(args.expect)} times for {len(args.hit)} --hit "
            "questions. Matched by position, so it has to line up: one each, "
            "or none at all.\nPass - for the ones you cannot name yet and "
            "their candidate rows will be printed with their ids."
        )
        return 1
    if not os.path.exists(args.db):
        print(f"{args.db} does not exist")
        return 1

    os.environ["RAG_DB_FILE"] = args.db
    from forge import rag
    from forge.config import RECALL_LEXICAL_MAX_DF, RECALL_MAX_DISTANCE

    cutoff = args.cutoff if args.cutoff is not None else RECALL_MAX_DISTANCE
    fractions = args.max_df or [RECALL_LEXICAL_MAX_DF]

    conn = rag.get_connection()
    if not rag.has_lexical_index(conn):
        print("this store has no lexical index and this SQLite has no FTS5.")
        return 1

    expects = args.expect or [None] * len(args.hit)
    questions = [
        (q, None if e in (None, "-") else e, True) for q, e in zip(args.hit, expects)
    ] + [(q, None, False) for q in args.miss]

    total = rag.count_entries(conn)["total"]
    print(f"--- {args.db}: {total} entries, cutoff {cutoff}")
    print("    vector: rank + distance, CUT when the cutoff would drop it.")
    print("    word:   rank + bm25 (lower is better). No cutoff applies here.\n")

    rescued: list[str] = []
    intruders: list[str] = []
    wrong_entry: list[str] = []

    for question, expect, is_hit in questions:
        print(f"{'hit ' if is_hit else 'miss'}  {question}")
        vector = (
            [] if args.no_vector else rag.search(conn, query=question, top_k=args.top_k)
        )
        if not args.no_vector:
            print(f"      vector      {_vector_cell(vector, expect, cutoff)}")

        for fraction in fractions:
            terms, too_common = rag.informative_terms(conn, question, max_df=fraction)
            lexical = rag.search_lexical(
                conn, query=question, top_k=args.lexical_top_k, max_df=fraction
            )
            print(
                f"      word {fraction:<5}  {_lexical_cell(lexical, expect)}"
                f"   terms: {', '.join(terms) or '(none admitted)'}"
            )
            if too_common:
                shown = ", ".join(f"{t}({df})" for t, df in too_common[:6])
                print(f"                    too common: {shown}")

            if not lexical:
                continue

            if is_hit and expect is not None:
                # --no-vector means the other channel was never asked,
                # so nothing here knows whether it would have
                # delivered. Counting a rescue against a channel that
                # did not run is how a harness reports a win it never
                # measured.
                if (
                    not args.no_vector
                    and not _delivered(vector, expect, cutoff)
                    and _row(lexical, expect)[0]
                ):
                    rescued.append(f"{question}  (max_df {fraction})")
                if lexical[0].get("id") is not None and str(lexical[0]["id"]) != str(
                    expect
                ):
                    wrong_entry.append(f"{question}  (max_df {fraction})")
                    _print_candidates(lexical, "word channel returned, in order:")
            elif not is_hit:
                intruders.append(f"{question}  (max_df {fraction})")
                _print_candidates(lexical, "word channel answered a MISS with:")
            else:
                _print_candidates(
                    lexical, "candidates -- name one with --expect next run:"
                )
        print()

    print("--- what the word channel changed")
    if args.no_vector:
        print("    (--no-vector: REACHED is not counted -- the other channel")
        print("     never ran, so nothing here knows what it would have given)")
    for label, rows, note in (
        ("REACHED", rescued, "the vector channel could not deliver these"),
        ("WRONG ENTRY", wrong_entry, "reached, but something else came first"),
        ("INTRUDERS", intruders, "a correct refusal turns into an answer"),
    ):
        print(f"  {label:<12} {len(rows):<3} -- {note}")
        for row in sorted(set(rows)):
            print(f"       {row}")

    if intruders and not rescued:
        print(
            "\n  Cost and no benefit on these questions. Leave RECALL_LEXICAL off,\n"
            "  or lower --max-df until the intruders stop, then check the\n"
            "  hits still reach."
        )
    elif rescued and not intruders:
        print(
            "\n  Wins and no cost ON THESE QUESTIONS, which is as far as this\n"
            "  goes: three misses is a sample, not an absence of misses."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
