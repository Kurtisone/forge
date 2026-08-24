#!/usr/bin/env python3
"""
What does burying a sentence in a compacted block cost, in distance?

This is the measurement the intake change stands on, and it had never
been made directly. It was inferred, correctly but indirectly, from
two numbers taken against the real store on 2026-08-22: the question
"Quels outils as-tu accès ?" sat at 0.9386 from a history_summary that
contained it nearly word for word, while "Le serveur de test tourne
sur quel port ?" sat at 0.671 from a short fact. Two different
questions, two different entries -- a comparison with two variables
moving at once.

Here only ONE thing moves. The same sentence is planted three ways and
asked the same question:

    alone     the sentence as its own entry -- the best case
    buried    a transcript containing it, stored as ONE entry, which
              is what compaction did until now
    split     the same transcript, stored one entry per exchange,
              which is what compaction does now

The number that matters is buried minus split. That is what the change
buys, and if it is not there, the change is wrong.

WHY IT SHOULD BE THERE (the mechanism, so a null result is readable)

The buried transcript is longer than EMBEDDING_MAX_CHARS, so rag._embed
splits it, embeds each chunk, and AVERAGES the chunk vectors into one.
The mean of a dozen unrelated subjects is close to no question in
particular. `split` makes the same number of embedding requests -- it
just keeps the results apart instead of collapsing them.

A null result would mean the embedding model is not sensitive to
length here and the dilution is coming from somewhere else. Say so
rather than shipping the change on faith.

THE NUMBER IS A LOWER BOUND

The block planted here is a few thousand characters, a handful of
chunks. A block evicted by a real compaction pass -- 59 messages on
2026-08-19 -- is an order of magnitude longer, so its single vector is
the mean of proportionally more unrelated subjects. Whatever cost this
harness prints, the store pays more.

RUNNING IT

Needs the embedding server (forge-embedding), nothing else.

    bench/in_container.sh rag_dilution

The six-command copy dance lives in bench/in_container.sh now -- the
faults in it are silent (a merged /tmp/arm holding two checkouts, a
harness pointed at the real store) and it was duplicated across every
file here.

It writes to a SEPARATE database (--db, default /tmp/rag_dilution.db)
and never touches data/forge_rag.db, for the same reason
bench/recall_distance.py does not: planting fixtures in the real store
leaves them there for the next real recall.
"""

from __future__ import annotations

import argparse
import os
import sys

# The sentence under test, and the question asked about it. Phrased so
# the question is not a copy of the sentence -- matching a paraphrase
# is the job, matching a copy is not.
NEEDLE = (
    "user: sur quel port tourne le serveur de test ?\n"
    "assistant: il écoute sur le 8080, c'est réglé dans le compose"
)
QUESTION = "Le serveur de test utilise quel numéro de port ?"

# Filler in the register the real store is full of: French, technical,
# about this project, and about anything BUT the needle. Filler that
# happened to mention ports would measure something else.
FILLER: list[tuple[str, str]] = [
    (
        "tu peux me rappeler pourquoi on est passés à podman ?",
        (
            "pas de démon root, et les conteneurs rootless marchent sans configuration "
            "supplémentaire sur le Deck"
        ),
    ),
    (
        "le proxy D-Bus, il tourne dans le conteneur ou sur l'hôte ?",
        (
            "sur l'hôte, en unité utilisateur systemd ; le conteneur ne fait que s'y "
            "connecter par le socket monté"
        ),
    ),
    (
        "pourquoi le prefill est si lent au premier tour ?",
        (
            "le cache KV de llama-server est vide après un redémarrage, donc tout le "
            "prompt routeur est recalculé, environ dix millisecondes par token"
        ),
    ),
    (
        "on garde ARCHITECTURE.md à la racine ?",
        "oui, c'est un engagement plutôt que de la doc sur le code, comme SECURITY.md",
    ),
    (
        "la grammaire GBNF accepte les underscores dans les noms de règles ?",
        (
            "non, llama.cpp lexe les noms avec is_word_char qui ne prend que lettres, "
            "chiffres et tirets, donc tout est en tirets"
        ),
    ),
    (
        "qu'est-ce qui a cassé la résolution DNS des conteneurs ?",
        (
            "un dns: nu dans le compose remplace tout le resolv.conf, il faut la forme à "
            "deux serveurs sinon les noms de conteneurs ne résolvent plus"
        ),
    ),
    (
        "pourquoi les commits doivent rester à ton nom ?",
        (
            "une question de présentation du dépôt, pas d'ego ; l'identité git est fixée "
            "avant chaque format-patch"
        ),
    ),
    (
        "le tiroir, ça correspond à quoi côté code ?",
        (
            "les messages marqués pinned dans l'historique, que la compaction n'évince "
            "jamais"
        ),
    ),
    (
        "on a tranché pour les tags git ?",
        (
            "on ne rétro-tague pas les versions passées, on pose un tag sur l'état "
            "courant après le merge et le trou reste visible"
        ),
    ),
    (
        "l'agent sysadmin peut redémarrer un service ?",
        (
            "non, il propose seulement ; la lecture des logs passe par un proxy read-only "
            "et rien n'est appliqué automatiquement"
        ),
    ),
    (
        "pourquoi /no_think reste dans les prompts des graphes ?",
        (
            "mesuré deux fois : sans lui, recall recopie l'exemple GOOD ANSWER au lieu de "
            "répondre, de façon déterministe"
        ),
    ),
    (
        "le job de délégation survit à un redémarrage ?",
        "il passe en interrupted au démarrage suivant, jamais de reprise automatique",
    ),
    (
        "pourquoi les patches plutôt qu'un bundle git ?",
        (
            "depuis la réattribution des auteurs en local les hashes ont changé, donc un "
            "bundle ne s'applique plus proprement sur le dépôt"
        ),
    ),
    (
        "le tripwire au démarrage, il sert à quoi exactement ?",
        (
            "à dire tout haut que files et test sont actifs ensemble, parce que cette "
            "combinaison élargit ce qu'un conteneur joignable peut exécuter"
        ),
    ),
    (
        "on met le seuil de compaction en messages à combien ?",
        (
            "il est à 80 aujourd'hui, mais c'était un proxy du budget en tokens à une "
            "époque où les tokens n'étaient pas mesurés"
        ),
    ),
    (
        "l'action edit du tool files, pourquoi elle existe ?",
        (
            "parce que le modèle ne chaîne pas read puis write de façon fiable, donc le "
            "remplacement se fait en un seul dispatch"
        ),
    ),
    (
        "qu'est-ce qui empêche une page web de choisir où on écrit ?",
        (
            "la garde de provenance : le chemin est lu dans la décision de routage, "
            "jamais dans la sortie d'un outil"
        ),
    ),
    (
        "pourquoi la découverte sysadmin remonte cinq cents unités ?",
        (
            "elle liste tout ce que busctl expose, y compris les .device, mais cette "
            "liste n'entre pas dans le prompt de synthèse"
        ),
    ),
    (
        "le badge de langage dans les blocs de code, il vient d'où ?",
        (
            "du tag de la fence markdown, affiché tel quel sans table de jolis noms, "
            "positionné en absolu dans le pre"
        ),
    ),
    (
        "on a une CI ?",
        (
            "ruff check, ruff format --check et pytest sur python 3.12, avec les deux "
            "fichiers de requirements"
        ),
    ),
    (
        "à quoi sert la jauge dans l'en-tête ?",
        (
            "la barre est une prévision calculée sans appeler le modèle, l'infobulle "
            "donne les compteurs mesurés du dernier tour"
        ),
    ),
    (
        "pourquoi ne pas relancer le compose avec force-recreate ?",
        (
            "ça recharge llama-server et vide le cache KV, donc le tour suivant repart "
            "sur un prefill à froid"
        ),
    ),
]

_MIN_USEFUL_GAIN = 0.05


def _rank_of_needle(hits: list[dict]) -> int:
    """
    Where the entry HOLDING the sentence lands, 1-based, or 0 if it is
    not in the list at all.

    Rank and distance answer different questions and both are needed.
    In the split shape the needle competes with a dozen sibling
    exchanges, so a good distance at rank 7 would mean the cut helped
    the vector and hurt the retrieval -- which the distance alone would
    hide.
    """
    for i, hit in enumerate(hits, start=1):
        if NEEDLE in hit["content"]:
            return i
    return 0


def _needle_distance(hits: list[dict]) -> float | None:
    for hit in hits:
        if NEEDLE in hit["content"]:
            return hit["distance"]
    return None


def _transcript(needle_at: int) -> list[str]:
    """The filler as exchanges, with the needle inserted in the middle."""
    blocks = [f"user: {q}\nassistant: {a}" for q, a in FILLER]
    blocks.insert(needle_at, NEEDLE)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="/tmp/rag_dilution.db")
    parser.add_argument("--question", default=QUESTION)
    args = parser.parse_args()

    if args.db.endswith("forge_rag.db"):
        print("refusing to plant fixtures in the real store")
        return 1

    # Set before importing forge.rag: RAG_DB_FILE is read at import.
    os.environ["RAG_DB_FILE"] = args.db
    if os.path.exists(args.db):
        os.remove(args.db)

    from forge import rag
    from forge.config import EMBEDDING_MAX_CHARS

    blocks = _transcript(len(FILLER) // 2)
    joined = "\n".join(blocks)

    conn = rag.get_connection()
    try:
        print(f"question : {args.question!r}\n")
        print(
            f"bloc de {len(blocks)} échanges, {len(joined)} caractères "
            f"-- soit {-(-len(joined) // EMBEDDING_MAX_CHARS)} chunks moyennés "
            f"en un seul vecteur dans la forme 'buried'\n"
        )

        try:
            rag.remember(conn, kind="fact", content=NEEDLE, project="alone")
            rag.remember(conn, kind="history_summary", content=joined, project="buried")
            rag.remember_many(
                conn, kind="history_summary", contents=blocks, project="split"
            )
        except rag.EmbeddingError as e:
            print(f"embedding server unreachable ({rag.EMBEDDING_URL}): {e}")
            return 1

        total = rag.count_entries(conn)["total"]
        results = {}
        for shape in ("alone", "buried", "split"):
            hits = rag.search(conn, args.question, top_k=total, project=shape)
            if not hits:
                print(f"{shape}: no hit at all -- the store did not return the row")
                return 1
            distance = _needle_distance(hits)
            if distance is None:
                print(
                    f"{shape}: the entry holding the sentence was not returned "
                    f"at all ({len(hits)} hits) -- nothing to compare"
                )
                return 1
            results[shape] = (distance, _rank_of_needle(hits), len(hits))
    finally:
        conn.close()

    print(f"{'forme':<10}{'distance':>10}{'rang du bloc qui contient la phrase':>40}")
    labels = {
        "alone": "seule",
        "buried": "enfouie",
        "split": "découpée",
    }
    for shape in ("alone", "buried", "split"):
        distance, rank, n = results[shape]
        print(f"{labels[shape]:<10}{distance:>10.4f}{f'{rank}/{n}':>40}")

    gain = results["buried"][0] - results["split"][0]
    floor = results["alone"][0]
    print(f"\ncoût de l'enfouissement (buried - split) : {gain:+.4f}")
    print(f"plancher (la phrase seule)               : {floor:.4f}")

    if gain < _MIN_USEFUL_GAIN:
        print(
            "\nPAS DE GAIN. Le découpage ne rapproche pas la phrase de la "
            "question sur cette machine avec ce modèle d'embedding, donc la "
            "dilution vient d'ailleurs que de la moyenne des chunks. Ne pas "
            "défendre le changement avec ce chiffre."
        )
        return 0

    print(
        "\nGAIN CONFIRMÉ. Le découpage récupère l'essentiel de l'écart entre "
        "un bloc entier et la phrase seule ; c'est exactement ce que la "
        "compaction par échange achète sur les entrées à venir, et ce que "
        "deploy/rag_resplit.py achète sur celles déjà écrites."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
