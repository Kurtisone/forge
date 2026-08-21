#!/usr/bin/env python3
"""
What distance does a GOOD memory hit sit at, on this box, with this
embedding model?

Nothing in the repository knows. Every distance measured so far -- the
five rows behind the invented-causality answer of 2026-08-19, at 0.90
to 1.0015 -- came from a query with no good answer in the store. That
is a sample of misses. A cutoff placed from misses alone silences real
hits and does it invisibly, which is worse than the failure it fixes:
a wrong answer gets argued with, "je n'ai rien en mémoire" gets
believed.

So this harness measures the OTHER half. It plants entries it wrote
itself, asks questions it knows the answers to, and asks the same store
questions it knows are unanswerable. Both distributions printed side by
side, and the cutoff goes in the gap.

    podman exec forge sh -c 'rm -rf /tmp/arm && mkdir -p /tmp/arm'
    podman cp src forge:/tmp/arm/
    podman cp bench/recall_distance.py forge:/tmp/arm/
    podman exec -it forge python /tmp/arm/recall_distance.py

The rm -rf is not cosmetic: podman cp merges into an existing directory
instead of replacing it, so without it the run is a mix of checkouts.

It writes to a SEPARATE database (--db, default /tmp/recall_bench.db)
and never touches data/forge_rag.db. Planting fixtures in the real
store would leave them there for the next real recall, which is the
same class of mistake as a benchmark that writes to production.

READING IT
----------
The number that matters is the GAP: the worst planted hit versus the
best unanswerable query. A threshold belongs inside it, nearer the
miss side.

  wide gap (say hits under 0.6, misses above 0.9)
      set RECALL_MAX_DISTANCE between them and the invented-causality
      answer becomes "je n'ai rien d'assez proche".

  no gap -- planted hits land in the same band as the misses
      then there is no threshold to find, the embedding is not
      separating these texts at all, and a cutoff would be a coin
      flip dressed as a setting. The answer is upstream: the store
      holds history_summary pointers written by compaction and
      almost no facts, so recall is being asked to answer from
      material that never contained the answer. Fix the intake, not
      the filter.

The second outcome is the likelier one, and printing it clearly is
worth as much as the number.
"""

from __future__ import annotations

import argparse
import os
import sys

# Planted entries, and the questions they answer. Deliberately in the
# register real memory entries are written in -- short, factual,
# French, one thing each. Phrased so no question repeats its entry's
# wording: matching a paraphrase is the job, matching a copy is not.
FIXTURES: list[tuple[str, str, str]] = [
    (
        "decision",
        (
            "Le proxy podman read-only écoute sur "
            "/run/forge-podman-ro-proxy.sock et tourne sur l'hôte, pas dans "
            "le container."
        ),
        "Où est le socket du proxy podman ?",
    ),
    (
        "decision",
        (
            "Forge tourne sur le Steam Deck pour l'instant ; le NiPoGi n'est "
            "pas encore la cible de déploiement."
        ),
        "Sur quelle machine Forge tourne-t-il aujourd'hui ?",
    ),
    (
        "note",
        (
            "Le modèle utilisé est un Qwen3.5-9B quantisé en Q4_K_M, servi "
            "par llama-server."
        ),
        "Quel modèle de langage est utilisé ?",
    ),
    (
        "note",
        "Les commits doivent rester sous le nom de Kurtisone, pas celui de Claude.",
        "Sous quel nom faut-il commiter ?",
    ),
    (
        "todo",
        "Le seuil de compaction en tokens est à 6000 et la cible à 3000.",
        "À combien est réglé le seuil de compaction ?",
    ),
]

# Questions the store demonstrably cannot answer. Same language and
# register as the answerable ones -- a miss written in English or about
# an obviously alien topic would be easy to separate for reasons that
# have nothing to do with retrieval quality.
UNANSWERABLE: list[str] = [
    "Quelle est la recette de la tarte tatin ?",
    "Combien de temps dure le vol Paris-Tokyo ?",
    "Quel est le nom du chat de la voisine ?",
    "Quelles sont les règles du jeu de tarot à cinq ?",
    "Quel est le prix moyen d'un vélo électrique ?",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="/tmp/recall_bench.db")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # Set before importing forge.rag: RAG_DB_FILE is read at import.
    os.environ["RAG_DB_FILE"] = args.db
    if os.path.exists(args.db):
        os.remove(args.db)

    from forge import rag

    conn = rag.get_connection()
    try:
        print(f"planting {len(FIXTURES)} entries into {args.db}")
        for kind, content, _question in FIXTURES:
            rag.remember(conn, kind=kind, content=content, project=None)

        print("\n=== HITS (the answer is in the store) ===")
        hit_distances = []
        for _kind, content, question in FIXTURES:
            results = rag.search(conn, query=question, top_k=args.top_k)
            best = results[0] if results else None
            # Rank matters as much as distance: a planted entry that
            # comes back second, behind another planted entry, means
            # the cutoff is not the only thing that needs looking at.
            rank = next(
                (
                    i + 1
                    for i, r in enumerate(results)
                    if r.get("content", "").startswith(content[:40])
                ),
                None,
            )
            distance = best.get("distance") if best else None
            if isinstance(distance, float):
                hit_distances.append(distance)
            print(
                f"  {distance if distance is None else round(distance, 4):<8} "
                f"rank={rank}  {question}"
            )

        print("\n=== MISSES (nothing in the store answers this) ===")
        miss_distances = []
        for question in UNANSWERABLE:
            results = rag.search(conn, query=question, top_k=args.top_k)
            distance = results[0].get("distance") if results else None
            if isinstance(distance, float):
                miss_distances.append(distance)
            print(
                f"  {distance if distance is None else round(distance, 4):<8} "
                f"        {question}"
            )
    finally:
        conn.close()

    print("\n=== VERDICT ===")
    if not hit_distances or not miss_distances:
        print("  no distances came back -- is the embedding server up?")
        return 1

    worst_hit = max(hit_distances)
    best_miss = min(miss_distances)
    print(f"  worst hit : {worst_hit:.4f}")
    print(f"  best miss : {best_miss:.4f}")

    if best_miss <= worst_hit:
        print(
            "\n  NO GAP. The worst real hit is at least as far as the closest\n"
            "  unanswerable query, so no value of RECALL_MAX_DISTANCE\n"
            "  separates them. Do not set one: it would be a coin flip in a\n"
            "  config file. The problem is upstream -- what the store\n"
            "  contains, not how it is filtered."
        )
        return 0

    suggested = worst_hit + (best_miss - worst_hit) / 2
    print(
        f"\n  GAP of {best_miss - worst_hit:.4f}. A cutoff inside it separates\n"
        f"  these two sets. Midpoint: RECALL_MAX_DISTANCE={suggested:.2f}\n"
        "\n  Lean BELOW the midpoint, toward the hits, before shipping it:\n"
        "  these fixtures are five short clean facts, and the real store is\n"
        "  mostly long compaction summaries whose good hits will sit\n"
        "  further out than anything measured here."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
