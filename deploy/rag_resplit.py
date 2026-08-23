#!/usr/bin/env python3
"""
Re-slice the compaction blocks already in the store, one entry per
exchange.

Compaction now indexes an evicted window as one entry per exchange
instead of one entry per block. That fixes what gets written from here
on and does nothing at all for what is already written -- and on
2026-08-22 the real store was 16 entries of which 11 were whole
blocks. Leaving them means the store stays mostly made of the shape
the change exists to remove, and the next measurement against it
measures the old problem.

WHAT IT DOES

For every `history_summary` entry, forge.transcript.split() puts the
stored text through the SAME pipeline compaction now uses on a live
eviction -- parse, drop what is not conversation, cut at user turns,
drop the exchanges that answered nothing.

That last step removes text. It is the same trade already accepted for
pointers and router JSON: an exchange whose reply is one of Forge's
own refusals is a near-copy of its own question, and on 2026-08-22 one
of them outranked the real answer to that question by a wide margin.
An entry made ENTIRELY of such exchanges is reported and left in
place, with its id, because deleting a row nobody asked to delete is
not a migration's job.
An entry that comes back unchanged is left alone: it is already the
right shape, and rewriting it would burn an embedding call to produce
the same row with a new id. Anything else is replaced by its units.

The shared pipeline is the point. The first version of this script cut
the same way compaction did but skipped compaction's filtering, and
the 2026-08-22 migration wrote a dozen entries whose entire content is
"[59 messages précédents compactés -- voir mémoire vectorielle #12]",
plus some raw router JSON. Both paths now call one function.

Nothing else is touched. `fact`, `decision` and `todo` entries are
one statement each by construction; splitting them is not a thing that
makes sense.

ORDER, AND WHY

New entries are inserted first, the old one deleted after, both inside
the same run. A crash in between therefore leaves the block stored
twice -- as one blob and as its pieces -- which is visible in `!memory`
and fixable by deleting the blob. The other order loses the block
outright. Duplicated is recoverable, deleted is not.

RUNNING IT

It needs the embedding server: every new entry is a new vector, so a
store of 11 blocks cutting into ~90 exchanges makes ~90 embedding
requests. That is the same number _embed already made when it chunked
those blocks to average them, but paid again, once.

    podman exec forge sh -c 'rm -rf /tmp/arm && mkdir -p /tmp/arm'
    podman cp src forge:/tmp/arm/
    podman cp deploy/rag_resplit.py forge:/tmp/arm/
    podman exec -it forge python /tmp/arm/rag_resplit.py            # dry run
    podman exec -it forge python /tmp/arm/rag_resplit.py --apply

The rm -rf is not cosmetic: podman cp merges into an existing
directory instead of replacing it, so without it the run is a mix of
checkouts.

Dry run is the default and prints exactly what --apply would do. Take
a copy of data/forge_rag.db first; this rewrites rows in place and
there is no undo.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=os.getenv("RAG_DB_FILE", "data/forge_rag.db"),
        help="Store to migrate (default: the configured RAG_DB_FILE).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write. Without it, nothing is inserted or deleted.",
    )
    parser.add_argument(
        "--backup",
        metavar="PATH",
        help="Copy the database here before writing anything.",
    )
    parser.add_argument(
        "--kind",
        default="history_summary",
        help="Which kind to re-slice (default: history_summary).",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"{args.db} does not exist")
        return 1

    # Set before importing forge.rag: RAG_DB_FILE is read at import.
    os.environ["RAG_DB_FILE"] = args.db

    from forge import rag, transcript

    if args.apply and args.backup:
        shutil.copy2(args.db, args.backup)
        print(f"backup written to {args.backup}")

    conn = rag.get_connection()
    try:
        before = rag.count_entries(conn)
        print(f"before: {before['total']} entries {before['by_kind']}")

        # Snapshot first. The entries this creates carry the same kind,
        # so re-reading the table mid-run would hand back the pieces it
        # just wrote and try to split them again.
        targets = _all_of_kind(conn, args.kind)
        print(f"{len(targets)} {args.kind} entries to inspect\n")

        split_count = new_count = inert_count = 0
        refused: list[int] = []
        dropped_count = 0
        for entry in targets:
            units = transcript.split(entry["content"])
            # Asked of the same module that does the cutting, never
            # re-derived here: a reporting path with its own copy of
            # the rules is free to disagree with the path that writes.
            gone = transcript.split_dropped(entry["content"])
            dropped_count += len(gone)
            for unit in gone:
                head = unit[:70].replace("\n", " / ")
                print(f"#{entry['id']:>4}   dropped  {head}…")

            if not units:
                # Nothing worth indexing: a pointer to another entry,
                # or -- since the non-answer filter -- an exchange
                # where Forge declined to answer and nothing else.
                #
                # Reported, not deleted, either way. Removing a row
                # nobody asked to remove is not a migration's job, and
                # `!forget <id>` is one command away. The ids are
                # gathered so that decision can be made on a list
                # rather than by scrolling.
                inert_count += 1
                why = "que des non-réponses" if gone else "rien d'indexable"
                if gone:
                    refused.append(entry["id"])
                print(f"#{entry['id']:>4}    inert  {why}, laissée en place")
                continue
            if units == [entry["content"]]:
                continue

            split_count += 1
            new_count += len(units)
            head = entry["content"][:60].replace("\n", " / ")
            print(f"#{entry['id']:>4}  {len(units):>3} units  {head}…")

            if not args.apply:
                continue

            try:
                ids = rag.remember_many(
                    conn, kind=entry["kind"], contents=units, project=entry["project"]
                )
            except rag.EmbeddingError as e:
                # Stop rather than skip. An embedding server that just
                # went away will fail every remaining entry too, and a
                # half-migrated store with no record of where it
                # stopped is worse than one that was not started.
                print(f"\nembedding server unreachable: {e}")
                print("stopping here -- entries already migrated are committed")
                return 1

            if not ids:
                # Every unit was already in the store, under another
                # entry. Left in place all the same: this script does
                # not delete rows on a judgement about redundancy.
                print("        -> every unit already stored, left in place")
                continue

            rag.forget(conn, entry["id"])
            print(f"        -> #{ids[0]}-#{ids[-1]}, #{entry['id']} removed")

        after = rag.count_entries(conn)
        if dropped_count:
            print(
                f"\n{dropped_count} units answered nothing and were left out "
                "(see forge/non_answer.py)"
            )
        if refused:
            ids = " ".join(f"#{i}" for i in refused)
            print(f"{len(refused)} entries are a refusal and nothing else: {ids}")
            print("nothing was deleted -- `!forget <id>` if you want them gone")
        if inert_count:
            print(f"\n{inert_count} entries hold nothing indexable (see !forget)")
        print(
            f"\n{split_count} entries would become {new_count}"
            if not args.apply
            else f"\n{split_count} entries became {new_count}"
        )
        print(f"after:  {after['total']} entries {after['by_kind']}")
        if not args.apply:
            print("\ndry run -- nothing was written. Re-run with --apply.")
    finally:
        conn.close()

    return 0


def _all_of_kind(conn, kind: str) -> list[dict]:
    """Every entry of one kind, oldest first, paging through
    rag.list_entries so the migration does not silently stop at its
    default limit."""
    from forge import rag

    out: list[dict] = []
    page = 200
    offset = 0
    while True:
        batch = rag.list_entries(conn, kind=kind, limit=page, offset=offset)
        out.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return list(reversed(out))


if __name__ == "__main__":
    sys.exit(main())
