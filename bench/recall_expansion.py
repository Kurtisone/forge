#!/usr/bin/env python3
"""
Does asking the same question in other words find the entry that the
question missed -- and what does it let in on the way?

WHAT THIS MEASURES, AND WHY IT IS NOT A GAP

bench/recall_distance.py places a threshold and bench/instruct_prefix.py
reads a direction, both from the same number: worst hit versus best
miss. That number is the wrong one here, because the expansion does
not run on every question. It runs on ONE path -- after the cutoff
has dropped everything -- so what it changes is not the distribution
of distances but the OUTCOME of the handful of questions that were
about to be refused.

So the verdict is counted in questions, against a cutoff:

    rescued        the expected entry was beyond the cutoff, the
                   rephrasings brought it within. This is the win.
    out of reach   the expected entry was beyond the cutoff, but
                   something ELSE was within it, so the rescue never
                   fires. Nothing here can fix that -- see below.
    already found  the expected entry was within the cutoff to begin
                   with. The expansion costs these nothing and does
                   nothing for them.
    false rescue   a --miss that was correctly refused, and is not
                   refused any more. This is the price, and it is the
                   number that decides whether to ship the mode on.

OUT OF REACH IS THE INTERESTING ROW

The rescue only runs when EVERY row of the first pass was dropped. A
question whose answer sits at 0.95 while an unrelated entry sits at
0.86 never reaches it: something was close enough, synthesis ran on
the wrong material, and the user got a fluent wrong answer instead of
a refusal. That was already true before this lot and it stays true
after it. If this row is large on your store, the expansion is not
what you are missing.

    bench/in_container.sh recall_expansion --db /tmp/real_copy.db \\
        --hit "Tu peux me lister mon matériel ?" --expect 308 \\
        --miss "Comment s'appelle mon chat ?"

NAME THE ENTRIES. Without --expect there is nothing to say the
rephrasing found the RIGHT row, and this store holds archived
exchanges that contain the question and outrank the answer to it --
the fault that produced the finest number bench/recall_distance.py
ever printed (2026-08-22, 0.4519 against a refusal). The hit column
falls back to the closest row and every verdict about hits is
suppressed, loudly.

--mode llm SPENDS A MODEL CALL PER QUESTION. On the Deck that is the
slow part of this harness by a wide margin; --mode terms is free and
answers a different question (how much of the failure was phrasing
rather than vocabulary). Both columns are printed by default because
the whole point is telling those two apart -- a deterministic rewrite
that is credited for the model's result is how a free mechanism gets
believed in.
"""

from __future__ import annotations

import argparse
import os
import sys

from _harness import placeholders, read_row, split_misplaced

_MODES = ("terms", "llm")


def _cell(row: tuple[float | None, int | None, float | None]) -> str:
    """scored (rank) [closest] -- the three numbers, in one column."""
    scored, rank, closest = row
    if scored is None:
        return f"{'--':<22}"
    rank_text = f"r{rank}" if rank else "r?"
    return f"{f'{scored:.4f} {rank_text:<3} [{closest:.4f}]':<22}"


def _within(distance: float | None, cutoff: float) -> bool:
    return isinstance(distance, float) and distance <= cutoff


def _closest_within(results: list[dict], cutoff: float) -> bool:
    """Would the cutoff have kept anything at all from this list?"""
    return any(_within(r.get("distance"), cutoff) for r in results)


def _expected_within(results: list[dict], expect: str | None, cutoff: float) -> bool:
    for row in results:
        if expect is not None and str(row.get("id")) == str(expect):
            return _within(row.get("distance"), cutoff)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        choices=[*_MODES, "both"],
        help="terms | llm | both (default: both).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help=(
            "The RECALL_MAX_DISTANCE to score against. Defaults to the one "
            "this container is configured with; there is no sensible fallback "
            "if neither is set, because the rescue pass is defined by it."
        ),
    )
    parser.add_argument("--hit", action="append", default=[], metavar="QUESTION")
    parser.add_argument("--expect", action="append", default=[], metavar="ID")
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
            "questions. Matched by position: one each, or none at all."
        )
        return 1
    if not os.path.exists(args.db):
        print(f"{args.db} does not exist")
        return 1

    modes = list(_MODES) if not args.mode or "both" in args.mode else args.mode

    os.environ["RAG_DB_FILE"] = args.db
    from forge import expansion, rag
    from forge.config import RECALL_MAX_DISTANCE

    cutoff = args.cutoff if args.cutoff is not None else RECALL_MAX_DISTANCE
    if cutoff is None:
        print(
            "no cutoff to score against: pass --cutoff, or run this where\n"
            "RECALL_MAX_DISTANCE is set. The rescue pass is DEFINED by the\n"
            "cutoff -- it runs when the cutoff drops everything -- so without\n"
            "one there is no question to ask here."
        )
        return 1

    expects = args.expect or [None] * len(args.hit)
    conn = rag.get_connection()

    tallies = {mode: {k: 0 for k in ("rescued", "missed", "false")} for mode in modes}
    out_of_reach: list[str] = []
    already_found: list[str] = []
    miss_already_answered: list[str] = []
    ranked_rows: list[tuple[str, str | None, int | None]] = []

    try:
        print(f"store     : {args.db}")
        print(f"cutoff    : {cutoff}  (regime {rag.query_fingerprint()})")
        print(f"modes     : {', '.join(modes)}\n")

        header = f"{'':<40} {'BASELINE':<22}"
        for mode in modes:
            header += f" {mode.upper():<22}"
        print(header)
        print("=" * len(header))

        print("HITS  (distance to the --expect entry, rank, [closest row])")
        for question, expect in zip(args.hit, expects):
            base_results = rag.search(conn, query=question, top_k=args.top_k)
            base = read_row(base_results, expect)
            ranked_rows.append((question, expect, base[1]))
            line = f"  {question[:38]:<38} {_cell(base)}"

            fires = not _closest_within(base_results, cutoff)
            found_already = _expected_within(base_results, expect, cutoff)

            per_mode = {}
            for mode in modes:
                variants = expansion.variants(question, mode)
                results = (
                    rag.search_many(conn, queries=variants, top_k=args.top_k)
                    if variants
                    else []
                )
                per_mode[mode] = (variants, results)
                line += f" {_cell(read_row(results, expect))}"
            print(line)

            for mode in modes:
                variants, _ = per_mode[mode]
                print(f"       {mode:<6} {variants if variants else '(aucune)'}")

            if expect is None:
                continue
            if found_already:
                already_found.append(question)
                continue
            if not fires:
                out_of_reach.append(question)
                continue
            for mode in modes:
                _, results = per_mode[mode]
                if _expected_within(results, expect, cutoff):
                    tallies[mode]["rescued"] += 1
                else:
                    tallies[mode]["missed"] += 1

        print("\nMISSES  (nothing should come back within the cutoff)")
        for question in args.miss:
            base_results = rag.search(conn, query=question, top_k=args.top_k)
            base = read_row(base_results, None)
            line = f"  {question[:38]:<38} {_cell(base)}"

            fires = not _closest_within(base_results, cutoff)
            for mode in modes:
                variants = expansion.variants(question, mode)
                results = (
                    rag.search_many(conn, queries=variants, top_k=args.top_k)
                    if variants
                    else []
                )
                line += f" {_cell(read_row(results, None))}"
                if fires and _closest_within(results, cutoff):
                    tallies[mode]["false"] += 1
            print(line)
            if not fires:
                miss_already_answered.append(question)
    finally:
        conn.close()

    print("\n=== VERDICT ===")
    if not args.expect and args.hit:
        print("  No --expect given, so nothing here can say the rephrasing found")
        print("  the RIGHT row. This store holds archived exchanges containing")
        print("  the question, and they outrank the answer to it. Name the")
        print("  entries; the hit verdict is suppressed until you do.\n")

    outranked, absent = split_misplaced(ranked_rows)
    if absent:
        print("  /!\\ the named entry was not in the baseline results at all:")
        for question in absent:
            print(f"        {question}")
        print("      Which is exactly what the expansion is for -- read their")
        print("      columns above rather than the counts below.\n")
    elif outranked:
        print("  named entry came back, but not first, for:")
        for question in outranked:
            print(f"        {question}")
        print()

    if already_found:
        print(f"  already found ({len(already_found)}) -- the cutoff kept the right")
        print("  entry without help. The expansion never runs for these.")
    if out_of_reach:
        print(f"  OUT OF REACH ({len(out_of_reach)}):")
        for question in out_of_reach:
            print(f"        {question}")
        print("      The right entry is beyond the cutoff and something else is")
        print("      within it, so the rescue never fires and synthesis runs on")
        print("      the wrong material. No rephrasing reaches these. If this")
        print("      row is the big one, the expansion is not what is missing.")
    if miss_already_answered:
        print(f"  misses already answered ({len(miss_already_answered)}) -- something")
        print("      was within the cutoff before any expansion, so these are not")
        print("      the expansion's doing and it cannot make them worse.")

    print()
    for mode in modes:
        counts = tallies[mode]
        print(
            f"  {mode:<6} rescued {counts['rescued']} | still missed "
            f"{counts['missed']} | FALSE RESCUES {counts['false']}"
        )

    print(
        "\n  Read the last column first. A rescue that answers two questions and\n"
        "  breaks one refusal has not obviously paid for itself: a refusal is a\n"
        "  failure the user argues with, and a fluent wrong answer is one they\n"
        "  believe. recall.rescued names the variant behind each one in the log,\n"
        "  which is where a bad rescue gets attributed rather than guessed at."
    )
    if "llm" in modes and "terms" in modes:
        print(
            "\n  And compare the two columns before keeping the expensive one. If\n"
            "  terms rescues what llm rescues, the failure was phrasing, not\n"
            "  vocabulary, and the model call is buying latency and nothing else."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
