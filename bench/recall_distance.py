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
      set RECALL_MAX_DISTANCE between them, leaning ABOVE the midpoint
      -- see MEASURED below for why that direction, which the first
      version of this file got backwards.

  no gap -- planted hits land in the same band as the misses
      then there is no threshold to find, the embedding is not
      separating these texts at all, and a cutoff would be a coin
      flip dressed as a setting. The answer is upstream: the store
      holds history_summary pointers written by compaction and
      almost no facts, so recall is being asked to answer from
      material that never contained the answer. Fix the intake, not
      the filter.

MEASURED -- Steam Deck, Qwen3-Embedding-0.6B, 2026-08-21
--------------------------------------------------------

    hits    0.6943  0.7439  0.9226  0.9351  0.9402   (all rank 1)
    misses  1.1096  1.1708  1.2071  1.2197  1.2586

    worst hit 0.9402 | best miss 1.1096 | gap 0.1693

A clean gap, and it settles a reading that had been wrong twice. The
five rows behind the invented-causality answer of 2026-08-19 sat at
0.90 / 0.95 / 0.99 / 0.995 / 1.0015. Those were first called
"orthogonal, pure noise" (they are not -- see config.py), and then, more
carefully, "weak but unplaceable". Against this scale they are placed:
the two closest are inside the range a genuine hit occupies, and the
other three sit in the gap where nothing measured lives.

Which means -- and this is the finding, not the threshold -- A CUTOFF
WOULD NOT HAVE PREVENTED THAT ANSWER. Any value that keeps real hits
(above ~0.94) keeps four of those five rows. The distance filter is a
guard against the tarte-tatin class of question, where nothing in the
store is remotely relevant. It is not a guard against the store being
full of long compaction summaries that sit at middling distance from
every question ever asked. That remains the real problem and it is an
intake problem.

DIRECTION, corrected. The first version of this file said to lean
BELOW the midpoint "because the real store's good hits will sit further
out than these fixtures". That reason argues the opposite conclusion:
if real hits are further out, a lower cutoff cuts them. Lean ABOVE the
midpoint, and stay clearly under the closest measured miss.

These fixtures are five short, clean, single-fact entries. The real
store is not. Before trusting any value in production, re-run against a
COPY of the real database with --no-plant and your own questions:

    podman exec forge cp /app/data/forge_rag.db /tmp/real_copy.db
    podman exec -it forge python /tmp/arm/recall_distance.py \
        --db /tmp/real_copy.db --no-plant \
        --hit "une question dont tu SAIS que la réponse est dedans" \
        --miss "une question dont tu sais qu'elle n'y est pas"

MEASURED AGAINST THE REAL STORE -- same day, --no-plant
-------------------------------------------------------

    hit   "Tu peux me lister mon matériel"              0.8934
    miss  "la date d'anniversaire de mon cousin ?"      0.9891

One pair, so not a distribution -- but enough to show that the fixture
numbers do not transfer. The real store's MISS sits at 0.9891, well
below the fixture misses (1.1096 and up) and below the value the
fixture run suggested (1.05). A cutoff set from the fixtures would let
that miss straight through.

Which is the predicted effect, arriving: real entries are mostly long
compaction summaries, they sit at middling distance from every question
ever asked, and they compress the whole scale. Hits AND misses move
closer together (gap 0.096 against the fixtures' 0.169).

So the fixture run is a sanity check on the mechanism, not a source of
the number. The number comes from --no-plant against a copy of the real
store, with at least three questions on each side.

WHY THREE. The first --no-plant run of all used the placeholder text
from this file's own help output as its two questions -- "une question
dont tu SAIS que la réponse est dans ta mémoire" and its negative. Both
are the same French sentence about memory, and both landed at ~0.80.
The harness printed "GAP of 0.0087" and confidently suggested a cutoff.
It should have refused: two questions are not two distributions, and a
gap that narrow is noise wearing a decimal point. It refuses now.
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
# Below this, --no-plant refuses to run. See WHY THREE above.
_MIN_QUESTIONS = 3

# Below this, no threshold is suggested however clean the two sets look.
# Calibrated against the two real gaps measured on 2026-08-21: 0.169
# between clean fixtures, 0.096 against the real store. 0.0087 -- what
# two near-identical questions produced -- must not yield a number.
_MIN_USABLE_GAP = 0.05

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
    parser.add_argument(
        "--no-plant",
        action="store_true",
        help=(
            "Do not plant fixtures and do not delete the database -- query "
            "it as it stands. Use with --hit/--miss against a COPY of the "
            "real store. Never point this at data/forge_rag.db itself."
        ),
    )
    parser.add_argument(
        "--hit",
        action="append",
        default=[],
        metavar="QUESTION",
        help="A question you KNOW the store can answer. Repeatable.",
    )
    parser.add_argument(
        "--miss",
        action="append",
        default=[],
        metavar="QUESTION",
        help="A question you know the store cannot answer. Repeatable.",
    )
    args = parser.parse_args()

    # Set before importing forge.rag: RAG_DB_FILE is read at import.
    os.environ["RAG_DB_FILE"] = args.db

    if args.no_plant:
        if not os.path.exists(args.db):
            print(f"--no-plant given but {args.db} does not exist")
            return 1
        if len(args.hit) < _MIN_QUESTIONS or len(args.miss) < _MIN_QUESTIONS:
            print(
                f"--no-plant needs at least {_MIN_QUESTIONS} --hit and "
                f"{_MIN_QUESTIONS} --miss questions "
                f"(got {len(args.hit)} and {len(args.miss)}).\n"
                "\nTwo questions are not two distributions. The first run of\n"
                "this mode used one of each -- and both were the placeholder\n"
                "sentences from this file's help text, which are the same\n"
                "French sentence about memory twice. They landed 0.0087 apart\n"
                "and this harness printed a confident threshold anyway.\n"
                "\nUse questions you actually asked Forge, and know the answer\n"
                "to."
            )
            return 1
        hits = [(None, None, q) for q in args.hit]
        misses = list(args.miss)
    else:
        if os.path.exists(args.db):
            os.remove(args.db)
        hits = list(FIXTURES)
        misses = list(UNANSWERABLE)

    from forge import rag

    conn = rag.get_connection()
    try:
        if not args.no_plant:
            print(f"planting {len(FIXTURES)} entries into {args.db}")
            for kind, content, _question in FIXTURES:
                rag.remember(conn, kind=kind, content=content, project=None)
        else:
            print(f"querying {args.db} as it stands, planting nothing")

        print("\n=== HITS (the answer is in the store) ===")
        hit_distances = []
        for _kind, content, question in hits:
            results = rag.search(conn, query=question, top_k=args.top_k)
            best = results[0] if results else None
            # Rank matters as much as distance: a planted entry that
            # comes back second, behind another planted entry, means
            # the cutoff is not the only thing that needs looking at.
            rank = (
                next(
                    (
                        i + 1
                        for i, r in enumerate(results)
                        if r.get("content", "").startswith(content[:40])
                    ),
                    None,
                )
                if content
                else None
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
        for question in misses:
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

    gap = best_miss - worst_hit
    if gap < _MIN_USABLE_GAP:
        print(
            f"\n  GAP of only {gap:.4f} -- too tight to act on. A threshold\n"
            "  placed inside it would be sorting by noise, and every entry\n"
            "  near the boundary would fall on whichever side the wording of\n"
            "  the day happened to put it.\n"
            "\n  Either the questions are too alike to separate (check that\n"
            "  the misses are really about something else), or this store\n"
            "  genuinely does not distinguish them -- in which case the\n"
            "  subject is what goes INTO it, not how it is filtered."
        )
        return 0

    midpoint = worst_hit + (best_miss - worst_hit) / 2
    suggested = worst_hit + (best_miss - worst_hit) * 0.65
    print(
        f"\n  GAP of {best_miss - worst_hit:.4f}. A cutoff inside it separates\n"
        f"  these two sets.  midpoint {midpoint:.4f}  |  "
        f"suggested RECALL_MAX_DISTANCE={suggested:.2f}\n"
        "\n  ABOVE the midpoint, deliberately. Real entries are longer and\n"
        "  messier than any fixture, so their good hits sit further out than\n"
        "  the ones measured here -- which argues for MORE room above the\n"
        "  hits, not less. (The first version of this harness said the\n"
        "  opposite while giving this same reason.)\n"
        "\n  A cutoff removes the case where nothing in the store is remotely\n"
        "  relevant. It does not remove the case where several middling\n"
        "  entries get welded into an invented answer -- check where your\n"
        "  own bad runs actually sat before expecting it to."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
