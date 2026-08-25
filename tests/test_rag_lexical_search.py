"""
The lexical channel: which words it agrees to search for, and what it
returns when it searches for them.

The case that motivated it is the first test. #313 is stored on the
real box as "Steam Deck, SteamOS, conteneurs Podman" -- a fact written
as a keyword list, which is the shape the vector channel retrieves
worst and the shape a word index retrieves best.
"""

import pytest

from forge import rag

FAKE_DIM = 8


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(
        rag, "_VEC_SCHEMA", rag._VEC_SCHEMA.replace("1024", str(FAKE_DIM))
    )
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * FAKE_DIM)
    connection = rag.get_connection()
    yield connection
    connection.close()


def _fill(conn, contents: list[str]) -> list[int]:
    return [rag.remember(conn, kind="fact", content=c, project=None) for c in contents]


def test_a_telegraphic_fact_is_reachable_by_its_own_words(conn):
    ids = _fill(
        conn,
        [
            "Steam Deck, SteamOS, conteneurs Podman",
            "Le chat de la voisine est roux",
            "Rendez-vous chez le dentiste jeudi",
            "Il faut penser à arroser les plantes",
            "Les courses sont faites pour la semaine",
        ],
    )

    hits = rag.search_lexical(conn, "Sur quoi tournent mes conteneurs Podman ?")

    assert [h["id"] for h in hits][:1] == ids[:1]


def test_an_accented_word_is_found_without_its_accent(conn):
    _fill(conn, ["Matériel : NiPoGi AM06PRO, 32 Go de RAM"])

    assert rag.search_lexical(conn, "liste mon materiel")


def test_common_words_are_refused_and_the_search_abstains(conn):
    # Every entry carries "que" and "mon": a question made only of
    # those has nothing to distinguish anything by, and the honest
    # answer from this channel is silence rather than five rows that
    # share a stopword with the question.
    _fill(
        conn,
        [
            "Je pense que mon travail avance",
            "Je crois que mon vélo est cassé",
            "Il paraît que mon voisin déménage",
            "On dirait que mon café est froid",
            "Je sais que mon train part tard",
        ],
    )

    kept, too_common = rag.informative_terms(conn, "que mon")

    assert kept == []
    assert {t for t, _ in too_common} == {"que", "mon"}
    assert rag.search_lexical(conn, "que mon") == []


def test_a_rare_word_survives_alongside_common_ones(conn):
    _fill(
        conn,
        [
            "Je pense que mon processeur chauffe",
            "Je crois que mon vélo est cassé",
            "Il paraît que mon voisin déménage",
            "On dirait que mon café est froid",
            "Je sais que mon train part tard",
        ],
    )

    kept, too_common = rag.informative_terms(conn, "que mon processeur chauffe")

    assert kept == ["processeur", "chauffe"]
    assert {t for t, _ in too_common} == {"que", "mon"}


def test_digits_are_searchable_words(conn):
    # "5500U" and "32" are the most discriminating words a question
    # about this box can carry, and a letters-only tokeniser drops
    # every one of them.
    _fill(conn, ["Le NiPoGi a un Ryzen 5500U et 32 Go de RAM"])

    assert rag.search_lexical(conn, "c'est quoi le 5500U ?")


def test_lexical_rows_carry_no_distance(conn):
    """
    A row admitted by the word channel has never been measured against
    RECALL_MAX_DISTANCE, so it must not arrive carrying something that
    looks like one.
    """
    _fill(conn, ["Steam Deck, SteamOS, conteneurs Podman"])

    (hit,) = rag.search_lexical(conn, "conteneurs Podman")

    assert "distance" not in hit
    assert isinstance(hit["score"], float)
    assert hit["matched_terms"] == ["conteneurs", "podman"]


def test_an_fts_keyword_is_searched_as_a_word(conn):
    # "OR" and "NEAR" are FTS5 syntax unless they are quoted. An
    # unquoted one turns a search into a parse error, which would
    # arrive as an empty channel nobody could explain.
    _fill(conn, ["Le mode near est activé sur la borne"])

    assert rag.search_lexical(conn, "near")


def test_filters_apply_to_the_lexical_channel_too(conn):
    _fill(conn, ["Steam Deck, SteamOS, conteneurs Podman"])
    rag.remember(
        conn,
        kind=rag.ARCHIVED_KIND,
        content="user: parle-moi de Podman\nassistant: volontiers",
        project=None,
    )

    # max_df=1.0: this test is about the filters, and on a two-row
    # store every word is in every row, so the admission rule (tested
    # above, on its own) would correctly refuse the search first.
    both = rag.search_lexical(conn, "Podman", max_df=1.0)
    facts = rag.search_lexical(
        conn, "Podman", exclude_kind=rag.ARCHIVED_KIND, max_df=1.0
    )

    assert len(both) == 2
    assert [r["kind"] for r in facts] == ["fact"]


def test_top_k_is_honoured(conn):
    _fill(conn, [f"Podman note numéro {i} sur le déploiement" for i in range(6)])

    hits = rag.search_lexical(conn, "Podman déploiement", top_k=2, max_df=1.0)

    assert len(hits) == 2


def test_an_empty_store_returns_nothing(conn):
    assert rag.informative_terms(conn, "Podman") == ([], [])
    assert rag.search_lexical(conn, "Podman") == []
