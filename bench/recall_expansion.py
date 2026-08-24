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
    wrong entry    the rescue FIRED on a hit and answered with
                   something that is not the named entry. The worst
                   outcome there is: a refusal replaced by a fluent
                   wrong answer, on a question whose answer exists.
    false rescue   a --miss that was correctly refused, and is not
                   refused any more. Same failure, from the side where
                   no right answer existed at all.

The last two are the price, and they are what decides whether to ship
a mode on. An intruder counted here also goes into the rescue regime
below, on the MISS side, because that is what it is: an entry a cutoff
for this pass would have to exclude.

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

AND IF YOU DO NOT KNOW THE ID, that is what the first run is for.
Pass `--expect -` for any question you cannot name yet, or leave
--expect off entirely, and the baseline rows are printed with their
ids and a slice of their content -- read them, pick the one that
actually answers, and run again naming it. Demanding an id the
harness itself refused to help you find was a real dead end on
2026-08-24, not a hypothetical one.

--repeat N ASKS EACH QUESTION N TIMES, INTERLEAVED. The expansion is a model call,
and on 2026-08-24 the same question produced different rewrites on two
consecutive runs -- 'processeur mémoire disque' once, 'processeur
mémoire écran' the next -- putting the same entry at 0.9766 and then
0.9435. Temperature is 0.0; greedy sampling on llama.cpp is still not
reproducible across cache states. That is a third of the gap the two
runs measured, so a threshold placed from a single draw is placed from
a coin toss. With --repeat, every question keeps its WORST draw:
farthest for a hit, nearest for a miss. A cutoff has to hold on a bad
day, not on the best one it was shown.

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

# The floor bench/recall_distance.py works to, for the same reason:
# one hit and one miss produce a gap, and a gap over two numbers is an
# anecdote with a decimal point on it.
_MIN_QUESTIONS = 3


def _cell(row: tuple[float | None, int | None, float | None]) -> str:
    """scored (rank) [closest] -- the three numbers, in one column."""
    scored, rank, closest = row
    if scored is None:
        return f"{'--':<22}"
    rank_text = f"r{rank}" if rank else "r?"
    return f"{f'{scored:.4f} {rank_text:<3} [{closest:.4f}]':<22}"


def _distance_of(results: list[dict], expect: str | None) -> float | None:
    """
    The named entry's distance, or None when it is not in the list.

    read_row falls back to the closest row so a table always has a
    number in it. That fallback is exactly wrong for a verdict: an
    entry that never came back is not an entry at a distance, and
    averaging one into a threshold is how a cutoff ends up measured
    against whatever happened to be nearby.
    """
    if expect is None:
        return None
    for row in results:
        if str(row.get("id")) == str(expect):
            distance = row.get("distance")
            return distance if isinstance(distance, float) else None
    return None


def _closest(results: list[dict]) -> float | None:
    distances = [r["distance"] for r in results if isinstance(r.get("distance"), float)]
    return min(distances) if distances else None


def _within(distance: float | None, cutoff: float) -> bool:
    return isinstance(distance, float) and distance <= cutoff


def _closest_within(results: list[dict], cutoff: float) -> bool:
    """Would the cutoff have kept anything at all from this list?"""
    return any(_within(r.get("distance"), cutoff) for r in results)


def _intruder(results: list[dict], expect: str | None, cutoff: float) -> dict | None:
    """
    The nearest row within the cutoff that is NOT the named entry.

    This is the outcome the counts had no name for. On 2026-08-24 the
    rescue put #307 at 0.9435 on 'Tu peux me lister mon matériel ?'
    while something else came in at 0.8777 -- inside the cutoff, ahead
    of it. In the deployment that question stops being refused and
    starts being answered, out of the wrong entry, and the harness
    printed FALSE RESCUES 0 because it only ever looked at the misses.
    """
    for row in sorted(results, key=lambda r: r.get("distance") or 0.0):
        if expect is not None and str(row.get("id")) == str(expect):
            continue
        if _within(row.get("distance"), cutoff):
            return row
    return None


def _expected_within(results: list[dict], expect: str | None, cutoff: float) -> bool:
    for row in results:
        if expect is not None and str(row.get("id")) == str(expect):
            return _within(row.get("distance"), cutoff)
    return False


def _worse(candidate: float | None, current: float | None, expect: str | None) -> bool:
    """
    Is this draw worse than the one already kept?

    Worse means FARTHEST when an entry is named -- the draw where the
    rescue is least likely to reach it -- and NEAREST when nothing is
    named, which is the miss side: the draw most likely to break a
    refusal. Two statements of one rule, from the two ends. A
    threshold read off the best draw holds until the next sampling.
    """
    if current is None:
        return True
    if expect is not None:
        return candidate > current
    return candidate < current


def _collect(
    conn, questions: list[tuple], modes: list[str], top_k: int, repeat: int
) -> dict:
    """
    Every question expanded `repeat` times, ROUND ROBIN, keeping each
    one's worst draw.

    The round robin is the whole point and it was not there in the
    first version. Asking the same question three times in a row on
    llama.cpp measures nothing: the first call warms the prompt cache
    and the next two are cache hits returning byte-identical output --
    measured 2026-08-24, prompt_ms 4116 then 189 then 186, same 107
    characters back each time.

    The variance that exists is BETWEEN cache states, not within one.
    Two separate runs of this harness, minutes apart, gave 'processeur
    mémoire disque' and then 'processeur mémoire écran' for the same
    question, moving the entry from 0.9766 to 0.9435. So a draw has to
    follow a DIFFERENT predecessor to be a different draw, which is
    what interleaving the questions gives: the cache in front of
    question three is question two's, not its own from a second ago.

    It still does not make the rescue deterministic. It makes the
    sampling honest, which is the only part a harness can fix.
    """
    from forge import expansion, rag

    best: dict[tuple, tuple[list[str], list[dict], float | None]] = {}
    for _ in range(max(1, repeat)):
        for key, question, expect in questions:
            for mode in modes:
                variants = expansion.variants(question, mode)
                results = (
                    rag.search_many(conn, queries=variants, top_k=top_k)
                    if variants
                    else []
                )
                measured = (
                    _distance_of(results, expect)
                    if expect is not None
                    else _closest(results)
                )
                # A draw that produced nothing counts, and counts as
                # the worst: it is what the deployment would have done
                # that time. Preferring the draws that produced
                # rewrites would measure a mechanism nobody runs.
                measured = float("inf") if measured is None else measured
                current = best.get((key, mode))
                if current is None or _worse(measured, current[2], expect):
                    best[(key, mode)] = (variants, results, measured)
    return best


def _print_candidates(results: list[dict]) -> None:
    """
    The rows themselves, with their ids, for a question nobody could
    name an entry for.

    The harness demanded an --expect id and offered no way to find
    one, which stopped a real measurement on 2026-08-24. These are the
    candidates: read them, pick the one that actually ANSWERS the
    question rather than the one that merely mentions it, and name it
    on the next run.
    """
    print("       candidates -- name one with --expect on the next run:")
    for row in results:
        distance = row.get("distance")
        shown = " ".join(str(row.get("content", "")).split())[:74]
        marker = f"{distance:.4f}" if isinstance(distance, float) else "  --  "
        print(
            f"         #{row.get('id'):<5} {marker}  {row.get('kind', ''):<16} {shown}"
        )


def _print_regime(
    mode: str, rows: dict[str, list[tuple[str, float]]], cutoff: float
) -> None:
    """
    The distances the RESCUE saw, which are not the ones the cutoff was
    measured against.

    A variant is a short phrase and the question it replaced was a
    sentence, so the whole distribution moves -- an entry a well-worded
    question finds at 0.73 can be the same entry a rephrasing finds at
    0.98. Scoring the rescue pass with RECALL_MAX_DISTANCE compares it
    against a scale it was not measured on, which is the fault
    docs/memory.md already names one section earlier about the
    threshold itself.

    So this prints what a cutoff FOR THIS PASS would have to separate,
    and refuses to name one on fewer than three questions a side.
    """
    hits, misses = rows["hits"], rows["misses"]
    if not hits and not misses:
        return

    print(f"\n  --- the rescue regime, {mode} ---")
    print("      distances the rephrasings saw, on the questions where the")
    print(f"      rescue actually fired. NOT the scale {cutoff} was measured on.")
    for question, distance in sorted(hits, key=lambda r: -r[1]):
        print(f"      hit    {distance:.4f}  {question[:52]}")
    for question, distance in sorted(misses, key=lambda r: r[1]):
        print(f"      miss   {distance:.4f}  {question[:52]}")

    if not hits or not misses:
        print("      (need both sides to say anything)")
        return

    worst_hit, best_miss = max(d for _, d in hits), min(d for _, d in misses)
    gap = best_miss - worst_hit
    print(f"      gap    {gap:+.4f}")

    if gap <= 0:
        print("      The two overlap: no cutoff for this pass admits the hits")
        print("      and refuses the misses. More rephrasings will not fix that;")
        print("      it is the rescue telling you it cannot be made safe here.")
    elif len(hits) < _MIN_QUESTIONS or len(misses) < _MIN_QUESTIONS:
        print(
            f"      Positive, on {len(hits)} hit(s) and {len(misses)} miss(es). "
            f"Not enough to place\n      a number -- {_MIN_QUESTIONS} a side, as "
            "recall_distance asks for, and for\n      the same reason: a gap over "
            "two numbers is an anecdote."
        )
    else:
        print(f"      A cutoff for this pass would sit above {worst_hit:.4f} and")
        print(
            f"      below {best_miss:.4f} -- midpoint {(worst_hit + best_miss) / 2:.4f},"
        )
        print("      and the room belongs above the hits, as it does for the")
        print("      first-pass threshold.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Ask each question this many times and keep the worst draw. The "
            "rewrites are not reproducible run to run; one draw is one coin "
            "toss."
        ),
    )
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
            "questions. Matched by position, so it has to line up: one each, or "
            "none at all.\n"
            "Pass - for the ones you cannot name yet and their candidate rows "
            "will be printed with their ids."
        )
        return 1
    if not os.path.exists(args.db):
        print(f"{args.db} does not exist")
        return 1

    modes = list(_MODES) if not args.mode or "both" in args.mode else args.mode

    os.environ["RAG_DB_FILE"] = args.db
    from forge import rag
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

    # "-" is "I do not know this one yet", which is a different thing
    # from "score it against whatever came back first". Both end up as
    # None, but the first prints the candidates.
    expects = [None if e.strip() in ("-", "") else e for e in args.expect] or [
        None
    ] * len(args.hit)
    conn = rag.get_connection()

    # Hits and misses in one list so the round robin can interleave
    # them: a draw only differs from the last one if a different
    # question came before it.
    questions = [("hit", i, question) for i, question in enumerate(args.hit)]
    questions += [("miss", i, question) for i, question in enumerate(args.miss)]
    plan = [
        (
            (kind, i),
            question,
            expects[i] if kind == "hit" else None,
        )
        for kind, i, question in questions
    ]

    tallies = {
        mode: {k: 0 for k in ("rescued", "wrong", "missed", "false")} for mode in modes
    }
    intruders: dict[str, list[tuple[str, dict]]] = {mode: [] for mode in modes}
    # Only questions where the rescue actually FIRES land here. The
    # rest are not in this regime at all.
    regime: dict[str, dict[str, list[tuple[str, float]]]] = {
        mode: {"hits": [], "misses": []} for mode in modes
    }
    out_of_reach: list[str] = []
    already_found: list[str] = []
    miss_already_answered: list[str] = []
    ranked_rows: list[tuple[str, str | None, int | None]] = []

    try:
        print(f"store     : {args.db}")
        print(f"cutoff    : {cutoff}  (regime {rag.query_fingerprint()})")
        print(f"modes     : {', '.join(modes)}")
        if args.repeat > 1:
            print(
                f"draws     : {args.repeat} per question, interleaved, worst one kept"
            )
        print()

        expanded = _collect(conn, plan, modes, args.top_k, args.repeat)

        header = f"{'':<40} {'BASELINE':<22}"
        for mode in modes:
            header += f" {mode.upper():<22}"
        print(header)
        print("=" * len(header))

        print("HITS  (distance to the --expect entry, rank, [closest row])")
        for hit_index, (question, expect) in enumerate(zip(args.hit, expects)):
            base_results = rag.search(conn, query=question, top_k=args.top_k)
            base = read_row(base_results, expect)
            ranked_rows.append((question, expect, base[1]))
            line = f"  {question[:38]:<38} {_cell(base)}"

            fires = not _closest_within(base_results, cutoff)
            found_already = _expected_within(base_results, expect, cutoff)

            per_mode = {}
            for mode in modes:
                variants, results, _ = expanded[(("hit", hit_index), mode)]
                per_mode[mode] = (variants, results)
                line += f" {_cell(read_row(results, expect))}"
            print(line)

            for mode in modes:
                variants, _ = per_mode[mode]
                print(f"       {mode:<6} {variants if variants else '(aucune)'}")

            if expect is None:
                _print_candidates(base_results)

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

                found = _distance_of(results, expect)
                if found is not None:
                    regime[mode]["hits"].append((question, found))

                # Counted whether or not the named entry also came
                # back: an intruder ahead of it decides the answer.
                gatecrasher = _intruder(results, expect, cutoff)
                if gatecrasher is not None:
                    tallies[mode]["wrong"] += 1
                    intruders[mode].append((question, gatecrasher))
                    distance = gatecrasher.get("distance")
                    if isinstance(distance, float):
                        regime[mode]["misses"].append(
                            (f"{question[:34]} -> #{gatecrasher.get('id')}", distance)
                        )

        print("\nMISSES  (nothing should come back within the cutoff)")
        for miss_index, question in enumerate(args.miss):
            base_results = rag.search(conn, query=question, top_k=args.top_k)
            base = read_row(base_results, None)
            line = f"  {question[:38]:<38} {_cell(base)}"

            fires = not _closest_within(base_results, cutoff)
            for mode in modes:
                _, results, _ = expanded[(("miss", miss_index), mode)]
                line += f" {_cell(read_row(results, None))}"
                if fires and _closest_within(results, cutoff):
                    tallies[mode]["false"] += 1
                nearest = _closest(results)
                if fires and nearest is not None:
                    regime[mode]["misses"].append((question, nearest))
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
            f"  {mode:<6} rescued {counts['rescued']} | WRONG ENTRY "
            f"{counts['wrong']} | still missed {counts['missed']} | "
            f"FALSE RESCUES {counts['false']}"
        )
        for question, row in intruders[mode]:
            distance = row.get("distance")
            marker = f"{distance:.4f}" if isinstance(distance, float) else "  --  "
            shown = " ".join(str(row.get("content", "")).split())[:56]
            print(f"           {question[:44]}")
            print(f"             answered with #{row.get('id')} at {marker}  {shown}")

    for mode in modes:
        _print_regime(mode, regime[mode], cutoff)

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
