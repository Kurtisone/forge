"""
The union of the two channels.

What matters here is not that both are called -- it is that a row
arrives tagged with the channel that admitted it, because the caller
applies a cutoff that belongs to exactly one of them.
"""

import pytest

from forge import rag

FAKE_DIM = 8

# Three vectors and not two, so the fixtures can hold the shape the
# real store has: the entry that answers the question is the FARTHEST
# thing from it, and a dozen unrelated entries sit between them. That
# is what "unreachable through the vector channel" means -- not that
# the distance is large, but that five other rows are nearer.
NEAR = [1.0] + [0.0] * (FAKE_DIM - 1)
MIDDLING = [0.7071, 0.7071] + [0.0] * (FAKE_DIM - 2)
FAR = [0.0, 1.0] + [0.0] * (FAKE_DIM - 2)


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(
        rag, "_VEC_SCHEMA", rag._VEC_SCHEMA.replace("1024", str(FAKE_DIM))
    )
    connection = rag.get_connection()
    yield connection
    connection.close()


def _store(conn, monkeypatch, content: str, vector: list[float]) -> int:
    monkeypatch.setattr(rag, "_embed", lambda text: vector)
    return rag.remember(conn, kind="fact", content=content, project=None)


def _fillers(conn, monkeypatch, n: int) -> None:
    for i in range(n):
        _store(conn, monkeypatch, f"Note anodine numéro {i} sans rapport", MIDDLING)


def test_an_entry_only_the_words_can_reach_comes_back(conn, monkeypatch):
    """
    The case the whole branch exists for. #313 is stored as a keyword
    list, so no query vector lands near it; its words are still right
    there.
    """
    _fillers(conn, monkeypatch, 12)
    telegram = _store(conn, monkeypatch, "Steam Deck, SteamOS, conteneurs Podman", FAR)

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(conn, "Où tournent mes conteneurs Podman ?")

    found = {h["id"]: h for h in hits}
    assert telegram in found
    assert found[telegram]["channel"] == "lexical"


def test_a_lexical_row_carries_no_distance(conn, monkeypatch):
    _fillers(conn, monkeypatch, 12)
    _store(conn, monkeypatch, "Steam Deck, SteamOS, conteneurs Podman", FAR)

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    (lexical,) = [
        h
        for h in rag.search_hybrid(conn, "conteneurs Podman")
        if h["channel"] == "lexical"
    ]

    # The caller's distance cutoff was measured on the vector channel.
    # A row that arrives without a distance cannot be judged by it,
    # and that is how the two admission rules stay apart.
    assert "distance" not in lexical


def test_a_row_both_channels_find_is_tagged_both(conn, monkeypatch):
    _fillers(conn, monkeypatch, 12)
    entry = _store(conn, monkeypatch, "Le NiPoGi porte un Ryzen 5500U", NEAR)

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(conn, "Quel Ryzen dans le NiPoGi ?")

    found = {h["id"]: h for h in hits}
    assert found[entry]["channel"] == "both"
    assert isinstance(found[entry]["distance"], float)
    assert isinstance(found[entry]["score"], float)


def test_vector_rows_are_ordered_before_word_matches(conn, monkeypatch):
    _fillers(conn, monkeypatch, 12)
    _store(conn, monkeypatch, "Steam Deck, SteamOS, conteneurs Podman", FAR)
    close = _store(conn, monkeypatch, "Une réponse sémantiquement proche", NEAR)

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(conn, "conteneurs Podman")

    assert hits[0]["id"] == close
    channels = [h["channel"] for h in hits]
    assert channels.index("vector") < channels.index("lexical")


def test_each_channel_keeps_its_own_budget(conn, monkeypatch):
    """
    A shared top_k sorted by distance would let vector rows crowd the
    word matches out -- and the case this exists for is the one where
    those vector rows are about to be cut by the cutoff anyway.
    """
    _fillers(conn, monkeypatch, 20)
    for i in range(4):
        _store(conn, monkeypatch, f"Conteneurs Podman, note {i}", FAR)

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(conn, "conteneurs Podman", top_k=2, lexical_top_k=3)

    assert sum(h["channel"] == "lexical" for h in hits) == 3


def test_a_question_of_common_words_falls_back_to_the_vector_channel(conn, monkeypatch):
    for i in range(12):
        _store(conn, monkeypatch, f"Je pense que mon dossier {i} avance", MIDDLING)

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(conn, "que mon")

    assert hits
    assert {h["channel"] for h in hits} == {"vector"}


def test_filters_reach_both_channels(conn, monkeypatch):
    _fillers(conn, monkeypatch, 12)
    _store(conn, monkeypatch, "Steam Deck, SteamOS, conteneurs Podman", FAR)
    monkeypatch.setattr(rag, "_embed", lambda text: FAR)
    rag.remember(
        conn,
        kind=rag.ARCHIVED_KIND,
        content="user: parle-moi des conteneurs Podman\nassistant: volontiers",
        project=None,
    )

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(conn, "conteneurs Podman", exclude_kind=rag.ARCHIVED_KIND)

    assert rag.ARCHIVED_KIND not in {h["kind"] for h in hits}


def test_the_word_channel_skips_archived_transcript(conn, monkeypatch):
    """
    Measured on the real store, 2026-08-25: every junk row the word
    channel returned was archived transcript, and both entries it
    rescued were facts.

    An archived unit contains the question verbatim -- compaction
    indexes one exchange per entry -- so for any question resembling
    one asked before, the transcript of that asking is the best word
    match in the store. #94 ("Tu peux analyser les logs de mon Steam
    Deck ? / Je ne peux pas...") beat the hardware fact on the
    hardware question that way.
    """
    _fillers(conn, monkeypatch, 12)
    fact = _store(conn, monkeypatch, "Matériel : NiPoGi, Ryzen 5500U", FAR)
    monkeypatch.setattr(rag, "_embed", lambda text: FAR)
    rag.remember(
        conn,
        kind=rag.ARCHIVED_KIND,
        content="user: Tu peux me lister mon matériel ?\nassistant: Je ne peux pas",
        project=None,
    )

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(conn, "Tu peux me lister mon matériel ?")

    word_rows = [h for h in hits if h["channel"] == "lexical"]
    assert [h["id"] for h in word_rows] == [fact]


def test_the_vector_channel_still_sees_archived_transcript(conn, monkeypatch):
    """
    The exclusion is the word channel's and nothing becomes
    unreachable: 0.88@a5c47b was measured with archived conversation
    in scope and that scope does not move.
    """
    _fillers(conn, monkeypatch, 12)
    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    archived = rag.remember(
        conn,
        kind=rag.ARCHIVED_KIND,
        content="user: Tu peux me lister mon matériel ?\nassistant: Je ne peux pas",
        project=None,
    )

    hits = rag.search_hybrid(conn, "Tu peux me lister mon matériel ?")

    assert archived in {h["id"] for h in hits}


def test_the_exclusion_can_be_turned_off_to_measure_the_other_side(conn, monkeypatch):
    _fillers(conn, monkeypatch, 12)
    monkeypatch.setattr(rag, "_embed", lambda text: FAR)
    archived = rag.remember(
        conn,
        kind=rag.ARCHIVED_KIND,
        content="user: Tu peux me lister mon matériel ?\nassistant: Je ne peux pas",
        project=None,
    )

    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    hits = rag.search_hybrid(
        conn, "Tu peux me lister mon matériel ?", lexical_exclude_archived=False
    )

    assert archived in {h["id"] for h in hits if h["channel"] == "lexical"}
