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

    podman exec forge sh -c 'rm -rf /tmp/arm && mkdir -p /tmp/arm'
    podman cp src forge:/tmp/arm/
    podman cp bench/instruct_prefix.py forge:/tmp/arm/
    podman exec forge cp /app/data/forge_rag.db /tmp/real_copy.db
    podman exec -it forge python /tmp/arm/instruct_prefix.py \\
        --db /tmp/real_copy.db \\
        --hit "Quel processeur a mon NiPoGi ?" --expect 308 \\
        --miss "Comment s'appelle mon chat ?"

Read-only. It never writes to the database it is given, but pass a
copy anyway -- the habit is what keeps a benchmark out of production.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# The task description Qwen's format expects. Theirs is written for
# web search; this one says what Forge actually stores, because the
# instruction is embedded along with the question and a description of
# the wrong corpus is a description of the wrong corpus.
DEFAULT_INSTRUCT = (
    "Given a question asked in conversation, retrieve the stored facts, "
    "decisions and past exchanges that answer it"
)

_PLACEHOLDER = re.compile(r"[<>]|\.\.\.|^\s*$|\bTODO\b|\bXXX\b")


def prefixed(instruct: str, query: str) -> str:
    """Qwen3-Embedding's query format. Documents get nothing."""
    return f"Instruct: {instruct}\nQuery: {query}"


def _row(results: list[dict], expect: str | None) -> tuple[float | None, int | None]:
    if not results:
        return None, None
    rank = None
    if expect is not None:
        rank = next(
            (i + 1 for i, r in enumerate(results) if str(r.get("id")) == str(expect)),
            None,
        )
    return results[0].get("distance"), rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--hit", action="append", default=[], metavar="QUESTION")
    parser.add_argument("--expect", action="append", default=[], metavar="ID")
    parser.add_argument("--miss", action="append", default=[], metavar="QUESTION")
    args = parser.parse_args()

    bad = [q for q in args.hit + args.miss if _PLACEHOLDER.search(q)]
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

    try:
        print(f"instruction: {args.instruct!r}\n")
        print(f"{'':<44} {'RAW':<18} {'PREFIXED':<18}")
        print("=" * 82)

        print("HITS")
        for question, expect in zip(args.hit, expects):
            r_raw = rag.search(conn, query=question, top_k=args.top_k)
            r_pre = rag.search(
                conn, query=prefixed(args.instruct, question), top_k=args.top_k
            )
            d_raw, k_raw = _row(r_raw, expect)
            d_pre, k_pre = _row(r_pre, expect)
            if isinstance(d_raw, float):
                raw_hits.append(d_raw)
            if isinstance(d_pre, float):
                pre_hits.append(d_pre)
            print(
                f"  {question[:42]:<42} "
                f"{d_raw:.4f} rank={k_raw!s:<5} "
                f"{d_pre:.4f} rank={k_pre!s:<5}"
            )
            if expect is not None and k_raw != k_pre:
                print(f"       #{expect} moved: rank {k_raw} -> {k_pre}")

        print("\nMISSES  (further is better here)")
        for question in args.miss:
            d_raw, _ = _row(rag.search(conn, query=question, top_k=args.top_k), None)
            d_pre, _ = _row(
                rag.search(
                    conn, query=prefixed(args.instruct, question), top_k=args.top_k
                ),
                None,
            )
            if isinstance(d_raw, float):
                raw_misses.append(d_raw)
            if isinstance(d_pre, float):
                pre_misses.append(d_pre)
            print(f"  {question[:42]:<42} {d_raw:.4f}{'':<12} {d_pre:.4f}")
    finally:
        conn.close()

    if not (raw_hits and raw_misses and pre_hits and pre_misses):
        print("\nno distances came back -- is the embedding server up?")
        return 1

    gap_raw = min(raw_misses) - max(raw_hits)
    gap_pre = min(pre_misses) - max(pre_hits)

    print("\n=== VERDICT ===")
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
