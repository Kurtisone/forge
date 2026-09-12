#!/usr/bin/env python3
"""
What the aggregation pass would fold, what it would refuse, and what
the block weighs on either side of it.

    bench/in_container.sh rag_aggregate --db /tmp/real_copy.db
    bench/in_container.sh rag_aggregate --db /tmp/real_copy.db --apply

NOTHING HERE CALLS A MODEL ANY MORE, and that is the finding this
harness produced. Seven calls across two runs on 2026-09-11 wrote zero
aggregates: the answer ran to the grammar's maximum item count every
time, `nipogi` as its three notes concatenated and `steam` as a cycle
of three repeated four times. The writer is arithmetic now, so the
whole pass is free, and the dry run shows the exact line each subject
would fold into rather than only the groups.

WITHOUT --apply IT WRITES NOTHING. It runs the real pass with the
write turned off -- the same code, not a second copy of it in a
harness -- and prints the line, the gate that would refuse it, and
what the block would weigh.

WITH --apply IT WRITES TO THE DATABASE YOU POINT IT AT. That is why
--db is required and has no default: point it at
/app/data/forge_rag.db and you are not measuring the pass, you are
running it on production. bench/in_container.sh puts a fresh copy at
/tmp/real_copy.db for exactly this reason.

WHAT THE COLUMNS MEAN

  SUBJECT     The entries one aggregate would speak for, and the word
              that named them. The name only matters for reading this
              output; the group is what the pass acts on.

  REFUSED     Which gate stopped it, if one did. `repetition` means
              two details say one thing in different words, which set
              arithmetic cannot merge -- the notes stay as they are.
              `budget` means the fold would not have been worth
              making. `closure`, `coverage` and `quorum` cannot fire
              under a writer that only copies details somebody wrote;
              seeing one is a finding about this module, not about
              the store.

  BLOCK       Entries and tokens before and after. This is the number
              the tier exists to move, and the one the hot tier's cap
              is measured against -- RECALL_HOT_MAX_TOKENS is a
              tripwire, and the aggregation pass is what is supposed
              to keep it one.

THERE IS DELIBERATELY NO SUGGESTED THRESHOLD, and no verdict. Every
number this prints is a description of one store. What it cannot tell
you is whether the lines are TRUE, and no harness can: read them.
Every word in them was typed by the person they describe, which is a
strong property and not that one -- a true detail and another true
detail can still be put side by side into a sentence nobody meant.
"""

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="WRITE the aggregates and the supersession links to --db",
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

    if args.apply:
        print(f"--- writing to {args.db}\n")
    report = aggregate.run_pass(
        conn, max_df=max_df, min_sources=min_sources, write=args.apply
    )

    for item in report:
        print(f"SUBJECT  {item['subject']}  {item['sources']}")
        if item.get("written"):
            verb = "wrote    " if item.get("id") else "would say"
            print(f"    {verb} {item['written']}")
        if item.get("refused"):
            print(f"    REFUSED   {item['refused']}")
            if item.get("invented"):
                print(f"              words from nowhere: {item['invented']}")
        if item.get("folded"):
            print(f"    FOLDED    {item['folded']} -> #{item['id']}")
        if item.get("folds"):
            print(f"    WOULD FOLD {item['folds']}")
        for source_id, missing in (item.get("held_back") or {}).items():
            print(f"    HELD BACK #{source_id}, missing {missing}")
        if item.get("tokens"):
            cost, saved = item["tokens"]
            print(f"    tokens    {saved} folded into {cost}")
        print()

    after = _block_after(rag.hot_entries(conn), report) if not args.apply else None
    entries_after = after if after is not None else rag.hot_entries(conn)
    after_t = tokens.estimate_tokens(hot_memory.render(entries_after))
    label = "WOULD BE" if not args.apply else "AFTER   "
    print(f"BLOCK BEFORE  {before_n} entries, {before_t} tokens")
    print(f"BLOCK {label}  {len(entries_after)} entries, {after_t} tokens")
    print(f"HEADROOM      {RECALL_HOT_MAX_TOKENS - after_t} tokens under the budget")
    print()
    print(
        "Read the lines. Every gate here is arithmetic on words, and none of "
        "them can tell you whether what would be written is true."
    )
    conn.close()
    return 0


def _block_after(entries: list[dict], report: list[dict]) -> list[dict]:
    """
    The block the reported folds would leave behind.

    The dry run changes nothing, so the only honest "after" is one
    computed from the report: the entries that would be superseded
    drop out, and the lines that would be written take their place,
    at the end, which is where rag.hot_entries reads a new row.
    """
    folded = {i for item in report for i in (item.get("folds") or [])}
    kept = [e for e in entries if e["id"] not in folded]
    written = [
        {"kind": "fact", "content": item["written"], "project": None}
        for item in report
        if item.get("folds")
    ]
    return kept + written


if __name__ == "__main__":
    raise SystemExit(main())
