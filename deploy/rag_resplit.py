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

For every `history_summary` entry, forge.transcript.split() cuts the
stored text back into the units transcript.blocks() would produce
today. An entry that yields one unit is left alone -- it is already
the right shape, and rewriting it would burn an embedding call to
produce the same row with a new id. An entry that yields several is
replaced by that many entries.

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

        split_count = new_count = 0
        for entry in targets:
            units = transcript.split(entry["content"])
            if len(units) < 2:
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

            rag.forget(conn, entry["id"])
            print(f"        -> #{ids[0]}-#{ids[-1]}, #{entry['id']} removed")

        after = rag.count_entries(conn)
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
