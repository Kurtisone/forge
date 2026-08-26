#!/usr/bin/env python3
"""
What the hot block answers, what it costs, and what it makes redundant.

    bench/in_container.sh rag_hot_tier --db /tmp/real_copy.db \\
        --hit "Tu peux me lister mon matériel ?" --expect 307 \\
        --hit "Sur quoi tournent mes conteneurs ?" --expect 313 \\
        --miss "Comment s'appelle mon chat ?"

WHAT THIS ONE ASKS THAT bench/rag_hybrid.py DOES NOT

rag_hybrid asks which channel REACHES an entry, because #307 and #313
were in the store and no phrasing brought them back. That question has
a trivial answer here and the harness says so rather than dressing it
up: the hot block carries every deliberate entry unconditionally, so
it reaches all of them, on every question, by construction. Printing
REACHED 11/11 would be reporting the definition.

The three questions worth measuring are different ones.

  COST        The tokens the block adds to every synthesis prompt,
              including the ones that already worked. On the Deck both
              LLM prompts share one llama-server slot and share no
              prefix, so this is prefilled at the full-recompute floor
              of 11.5-13.2 ms/token on EVERY recall -- the harness
              turns the token count into that range in seconds,
              because a token count is not a cost anyone feels.

  SUBSUMED    Rows the lexical channel returns that are ALREADY in the
              hot block. This is the number the branch turns on.
              RECALL_LEXICAL_EXCLUDE_ARCHIVED ships true, so the word
              channel only ever returns non-archived rows -- which is
              exactly the set the hot block already carries whole. If
              that intersection is total, then on this store the hot
              tier subsumes the measured gain of v3.17, and running
              both means paying for two mechanisms where one works.
              That is a real possibility and the harness is built to
              be able to report it rather than to avoid it.

  ADDS        Rows the lexical channel returns that the hot block does
              NOT hold. This is what would still justify running both.
              With --include-archived it is where archived transcript
              shows up, which is the other half of the same measurement.

  HEADROOM    How far the store is from RECALL_HOT_MAX_TOKENS. The cap
              is a tripwire and not a policy (see config.py); this is
              the number that says whether that is still true.

NO SUGGESTED THRESHOLD, for the same reason rag_hybrid prints none:
there is no number to calibrate here. The cap is a budget, not a
measurement, and the only thing that would change it is the store
growing -- which HEADROOM reports directly.

This harness makes NO LLM call and NO embedding call unless a --hit or
--miss is given; reading the block is a plain sqlite SELECT. It never
writes.
"""

import argparse
import os

from _harness import placeholders

#: The prefill floor measured on the Deck, ms per token, in the regime
#: where nothing is cached -- which is the regime this block lives in
#: today (see config.py, RECALL_HOT_FACTS).
_PREFILL_MS_PER_TOKEN = (11.5, 13.2)


def _cost_line(token_count: int) -> str:
    low, high = (token_count * ms / 1000 for ms in _PREFILL_MS_PER_TOKEN)
    return f"{token_count} tokens  ~{low:.1f}-{high:.1f} s of prefill per recall"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="override RECALL_HOT_MAX_TOKENS for this run",
    )
    parser.add_argument("--lexical-top-k", type=int, default=3)
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="let the word channel match archived transcript, to measure "
        "the other side of RECALL_LEXICAL_EXCLUDE_ARCHIVED",
    )
    parser.add_argument("--hit", action="append", default=[], metavar="QUESTION")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="ID",
        help="the entry id that should answer the --hit in the same position",
    )
    parser.add_argument("--miss", action="append", default=[], metavar="QUESTION")
    args = parser.parse_args()

    bad = placeholders(args.hit + args.miss)
    if bad:
        print("these look like unfilled placeholders, not questions:")
        for question in bad:
            print(f"  {question}")
        return 1
    if args.expect and len(args.expect) != len(args.hit):
        print(
            f"--expect given {len(args.expect)} times for {len(args.hit)} --hit "
            "questions. Matched by position, so it has to line up."
        )
        return 1
    if not os.path.exists(args.db):
        print(f"{args.db} does not exist")
        return 1

    os.environ["RAG_DB_FILE"] = args.db
    from forge import hot_memory, rag, tokens
    from forge.config import RECALL_HOT_MAX_TOKENS, RECALL_LEXICAL_MAX_DF

    budget = args.budget if args.budget is not None else RECALL_HOT_MAX_TOKENS

    conn = rag.get_connection()
    entries = rag.hot_entries(conn)
    if not entries:
        print(
            f"{args.db} holds no deliberate entries. The hot tier has nothing "
            "to carry, and that is the finding."
        )
        return 1

    kept, dropped = hot_memory.fit(entries, budget)
    body = hot_memory.render(kept)
    block_tokens = tokens.estimate_tokens(body)
    total = rag.count_entries(conn)
    archived = total["by_kind"].get(rag.ARCHIVED_KIND, 0)

    print(f"--- {args.db}: {total['total']} entries, {archived} archived")
    print(f"    hot block   {len(kept)} entries, {_cost_line(block_tokens)}")
    if dropped:
        print(
            f"    TRUNCATED   {dropped} entries did not fit in {budget}. The cap "
            "stopped being a tripwire -- that is the aggregation tier's work "
            "arriving, not a number to raise."
        )
    else:
        margin = budget - block_tokens
        print(
            f"    HEADROOM    {margin} tokens under the {budget} budget "
            f"(~{margin * len(kept) // max(block_tokens, 1)} more entries at this size)"
        )
    print()
    print(body)
    print()

    if not args.hit and not args.miss:
        print(
            "no --hit/--miss given, so nothing was searched and nothing was "
            "compared. Add them to measure SUBSUMED/ADDS against the word "
            "channel."
        )
        return 0

    if not rag.has_lexical_index(conn):
        print(
            "this store has no lexical index and this SQLite has no FTS5, "
            "so the comparison against the word channel cannot run."
        )
        return 1

    in_block = {e["id"] for e in kept}
    expects = args.expect or [None] * len(args.hit)
    questions = [
        (q, None if e in (None, "-") else e, True) for q, e in zip(args.hit, expects)
    ] + [(q, None, False) for q in args.miss]

    subsumed_total = 0
    adds_total = 0
    covered: list[str] = []
    uncovered: list[str] = []

    scope = "words match everything" if args.include_archived else "words skip archived"
    print(f"--- word channel vs the block ({scope})\n")

    for question, expect, is_hit in questions:
        lexical = rag.search_lexical(
            conn,
            query=question,
            top_k=args.lexical_top_k,
            max_df=RECALL_LEXICAL_MAX_DF,
            exclude_kind=(None if args.include_archived else rag.ARCHIVED_KIND),
        )
        ids = [r["id"] for r in lexical]
        subsumed = [i for i in ids if i in in_block]
        adds = [i for i in ids if i not in in_block]
        subsumed_total += len(subsumed)
        adds_total += len(adds)

        print(f"{'hit ' if is_hit else 'miss'}  {question}")
        print(f"      word returns  {ids or '(nothing)'}")
        print(f"      SUBSUMED      {subsumed or '(none)'}")
        print(f"      ADDS          {adds or '(none)'}")

        if expect is not None:
            where = "in the block" if int(expect) in in_block else "NOT in the block"
            (covered if int(expect) in in_block else uncovered).append(expect)
            print(f"      #{expect}        {where}")
        print()

    print(f"SUBSUMED {subsumed_total} / ADDS {adds_total}")
    if adds_total == 0 and subsumed_total:
        print(
            "Every row the word channel returned was already in the prompt. On "
            "this store the hot tier subsumes what RECALL_LEXICAL was measured "
            "to rescue, and running both pays twice for one mechanism."
        )
    if uncovered:
        print(
            f"NOT COVERED: {', '.join(uncovered)} -- an --expect the block does "
            "not hold is archived transcript, which is the cold tier's job and "
            "not this one's."
        )
    elif covered:
        print(f"COVERED: every --expect entry ({', '.join(covered)}) is in the block.")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
