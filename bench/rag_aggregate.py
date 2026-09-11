#!/usr/bin/env python3
"""
What the aggregation pass would fold, what it would refuse, and what
the block weighs on either side of it.

    bench/in_container.sh rag_aggregate --db /tmp/real_copy.db
    bench/in_container.sh rag_aggregate --db /tmp/real_copy.db --llm

WITHOUT --llm THIS MAKES NO MODEL CALL AND WRITES NOTHING. It reads the
deliberate entries, groups them by subject with the same arithmetic the
pass uses, and prints what it found. That half is free, deterministic,
and the half worth looking at first: if the grouping is wrong, nothing
downstream can be right.

WITH --llm IT WRITES TO THE DATABASE YOU POINT IT AT. One call per
subject, then every gate, then the entries and the supersession links.
That is why --db is required and has no default: point it at
/app/data/forge_rag.db and you are not measuring the pass, you are
running it on production. bench/in_container.sh puts a fresh copy at
/tmp/real_copy.db for exactly this reason.

WHAT THE COLUMNS MEAN

  SUBJECT     The entries one aggregate would speak for, and the word
              that named them. The name only matters for reading this
              output; the group is what the pass acts on.

  REFUSED     Which gate stopped it, if one did. `closure` means the
              sentence said something no entry says -- on llama.cpp
              the grammar makes that unsamplable, so seeing it here
              means the grammar was not in force. `coverage` is
              reported per source under HELD BACK, with the words that
              went missing. `quorum` and `budget` mean the fold would
              not have been worth making.

  BLOCK       Entries and tokens before and after. This is the number
              the tier exists to move, and the one the hot tier's cap
              is measured against -- RECALL_HOT_MAX_TOKENS is a
              tripwire, and the aggregation pass is what is supposed
              to keep it one.

THERE IS DELIBERATELY NO SUGGESTED THRESHOLD, and no verdict. Every
number this prints is a description of one store. What it cannot tell
you is whether the sentences are TRUE, and no harness can: read them.
"""

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="actually call the model and WRITE the result to --db",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=None,
        help="override COMPACTION_AGGREGATE_MAX_DF for this run",
    )
    parser.add_argument("--min-sources", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"{args.db} does not exist")
        return 1

    os.environ["RAG_DB_FILE"] = args.db
    from forge import aggregate, hot_memory, rag, tokens
    from forge.config import (
        COMPACTION_AGGREGATE_MAX_DF,
        COMPACTION_AGGREGATE_MIN_SOURCES,
        RECALL_HOT_MAX_TOKENS,
    )

    max_df = args.max_df if args.max_df is not None else COMPACTION_AGGREGATE_MAX_DF
    min_sources = (
        args.min_sources
        if args.min_sources is not None
        else COMPACTION_AGGREGATE_MIN_SOURCES
    )

    conn = rag.get_connection()
    entries = rag.hot_entries(conn)
    if len(entries) < min_sources:
        print(
            f"{args.db} holds {len(entries)} deliberate entries. There is "
            "nothing to aggregate, and that is the finding."
        )
        return 1

    corpus = rag.list_entries(conn, limit=aggregate._CORPUS_ENTRIES)
    freq = aggregate.frequencies(corpus)
    limit = aggregate.ceiling(len(corpus), max_df)
    before_n = len(entries)
    before_t = tokens.estimate_tokens(hot_memory.render(entries))

    print(f"--- {args.db}: {len(corpus)} entries, {before_n} of them deliberate")
    print(f"    BLOCK BEFORE  {before_n} entries, {before_t} tokens")
    print(f"    a word in more than {limit} entries identifies nothing here")
    print()

    found = aggregate.subjects(entries, freq, limit, min_sources)
    if not found:
        print(
            "No subject: no informative word names "
            f"{min_sources} deliberate entries at once. Nothing here overlaps, "
            "so there is nothing for this tier to do."
        )
        conn.close()
        return 0

    for subject in found:
        print(f"SUBJECT  {subject.term}  ({len(subject.ids)} entries: {subject.ids})")
        for entry in subject.entries:
            print(f"    #{entry['id']}  {entry['content']}")
        print()

    if not args.llm:
        print(
            "No --llm, so no sentence was written and nothing was changed. "
            "Read the groups above first: if they are wrong, nothing the "
            "model writes for them can be right."
        )
        conn.close()
        return 0

    print(f"--- writing to {args.db}\n")
    report = aggregate.run_pass(conn, max_df=max_df, min_sources=min_sources)

    for item in report:
        print(f"SUBJECT  {item['subject']}  {item['sources']}")
        if item.get("written"):
            print(f"    wrote     {item['written']}")
        if item.get("refused"):
            print(f"    REFUSED   {item['refused']}")
            if item.get("invented"):
                print(f"              words from nowhere: {item['invented']}")
        if item.get("folded"):
            print(f"    FOLDED    {item['folded']} -> #{item['id']}")
        for source_id, missing in (item.get("held_back") or {}).items():
            print(f"    HELD BACK #{source_id}, missing {missing}")
        if item.get("tokens"):
            cost, saved = item["tokens"]
            print(f"    tokens    {saved} folded into {cost}")
        print()

    after = rag.hot_entries(conn)
    after_t = tokens.estimate_tokens(hot_memory.render(after))
    print(f"BLOCK BEFORE  {before_n} entries, {before_t} tokens")
    print(f"BLOCK AFTER   {len(after)} entries, {after_t} tokens")
    print(f"HEADROOM      {RECALL_HOT_MAX_TOKENS - after_t} tokens under the budget")
    print()
    print(
        "Read the sentences. Every gate here is arithmetic on words, and "
        "none of them can tell you whether what was written is true."
    )
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
