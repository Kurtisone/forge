"""
Tests for the pass itself: what it writes, and the ways it refuses to.

The rule they share is the one this branch was opened on: NOTHING IS
WRITTEN UNLESS IT IS GOING TO REPLACE SOMETHING. Every gate is a
comparison between texts, so all of them run before the entry is
stored -- an aggregate that would be one more overlapping line in the
block instead of one fewer never reaches the store at all.

There is no model to stub any more, which changes what these tests
are. Three of the gates -- closure, coverage, quorum -- cannot fail
under a writer that only copies details somebody wrote, so they are
exercised here through a stubbed `merge`. That is not a contrived
setup: it is exactly the shape of the day somebody puts a writer back
in, and a gate nothing exercises is a gate nobody notices breaking.
"""

import pytest

from forge import aggregate, rag

NIPOGI = [
    "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go",
    "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible, services Podman",
    "Le NiPoGi a 32 Go de RAM",
    "Le NiPoGi AM06PRO a un processeur Ryzen 5500U et 32 Go de RAM",
]

#: The real store's `steam` subject, where the older note is the
#: newer one's own opening words.
STEAM = [
    "Possède un Steam Deck",
    ("Possède un Steam Deck sous SteamOS, fait tourner des conteneurs Podman dessus"),
]

#: Stands in for the archived half of a real store, so that `de`, `le`,
#: `a`, `et`, `un` and `avec` are common enough to identify nothing.
FILLER = [
    f"user: {q} assistant: Oui, et c'est une réponse avec un détail : "
    "le reste est de ce côté, il y a tout"
    for q in (
        "Tu peux me dire ce que tu sais ?",
        "Et le reste ?",
        "Merci",
        "Bonjour",
        "Ça marche ?",
        "On continue ?",
        "Tu as le fichier ?",
        "Un souci ?",
        "Et après ?",
        "C'est tout ?",
        "Redis-moi",
        "OK",
        "Tu confirmes ?",
        "Et la suite ?",
        "Tu peux relire ?",
        "C'est bon pour toi ?",
        "Tu as vu le message ?",
        "On garde ça ?",
        "Tu notes ?",
        "Et ensuite ?",
    )
]


def _store(path, monkeypatch, deliberate):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(path))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    for content in deliberate:
        rag.remember(conn, kind="fact", content=content, project=None)
    for content in FILLER:
        rag.remember(conn, kind="history_summary", content=content, project=None)
    return conn


@pytest.fixture
def store(tmp_path, monkeypatch):
    conn = _store(tmp_path / "rag.db", monkeypatch, NIPOGI)
    yield conn
    conn.close()


def _writes(monkeypatch, text):
    """Stand in for `merge`, for the gates its output can no longer fail."""
    monkeypatch.setattr(
        aggregate,
        "merge",
        lambda entries, term: aggregate.Merged(text, ()),
    )


def _run(conn, min_sources=2):
    return aggregate.run_pass(conn, max_df=0.2, min_sources=min_sources)


# --- What it writes --------------------------------------------------------


def test_the_block_goes_from_four_lines_to_one(store):
    report = _run(store)

    assert len(report) == 1
    assert len(report[0]["folded"]) == 4
    assert len(rag.hot_entries(store)) == 1


def test_the_line_it_writes_is_made_of_the_notes_own_details(store):
    _run(store)

    assert rag.hot_entries(store)[0]["content"] == (
        "Matériel : SSD 256 Go, Arch, Ansible, services Podman, "
        "Le NiPoGi AM06PRO a un processeur Ryzen 5500U et 32 Go de RAM."
    )


def test_it_writes_the_same_thing_on_a_second_run(store, tmp_path, monkeypatch):
    """
    The first property a writer with no model owes: two stores holding
    the same notes hold the same aggregate.
    """
    first = _run(store)[0]["written"]

    again = _store(tmp_path / "again.db", monkeypatch, NIPOGI)
    second = _run(again)[0]["written"]
    again.close()

    assert first == second


def test_the_sources_are_still_there_and_still_findable(store):
    _run(store)

    stored = {r[0] for r in store.execute("SELECT content FROM memory_entries")}
    assert set(NIPOGI) <= stored


def test_nothing_a_source_says_is_lost(store):
    """
    Coverage, as a property of the writer rather than a verdict on an
    answer. Every detail is either kept verbatim or dropped in favour
    of one that contains all of its words, so no informative word of a
    source can be missing from the line that replaces it.
    """
    report = _run(store)

    assert report[0].get("held_back") == {}


def test_the_line_can_never_be_longer_than_the_notes(store):
    """
    What the runaway guard used to catch. A writer that only copies
    details cannot run to n_predict, because there is no predicting:
    the longest line it can produce is every detail once.
    """
    report = _run(store)

    assert len(report[0]["written"]) < sum(len(n) for n in NIPOGI)


# --- The gates -------------------------------------------------------------


def test_a_word_from_nowhere_is_not_stored(store, monkeypatch):
    """
    Closure. Unreachable under `merge`, which is the point: this is
    the assertion that says the writer only copies, and it stands
    where a gate on a model used to.
    """
    _writes(monkeypatch, "Le NiPoGi AM06PRO a 32 Go de RAM et une carte Nvidia.")

    report = _run(store)

    assert report[0]["refused"] == "closure"
    assert "nvidia" in report[0]["invented"]
    assert len(rag.hot_entries(store)) == 4


def test_two_details_saying_one_thing_in_other_words_are_not_merged(
    tmp_path, monkeypatch
):
    """
    Repetition, and the only gate the arithmetic can still fail on its
    own output. `32 Go de RAM` and `32 Go de mémoire` share no word
    set and neither contains the other, so both survive deduplication
    and the line says one thing twice. Refusing leaves the block with
    the two overlapping entries it already had, which is the outcome
    this tier is supposed to improve on and not the one it is allowed
    to fake.
    """
    conn = _store(
        tmp_path / "synonym.db",
        monkeypatch,
        [
            "Matériel : NiPoGi AM06PRO, 32 Go de RAM, SSD 256 Go",
            "Le NiPoGi a 32 Go de mémoire, Arch",
        ],
    )

    report = _run(conn)

    assert report[0]["refused"] == "repetition"
    assert "32 go" in report[0]["repeated"]
    assert len(rag.hot_entries(conn)) == 2
    conn.close()


def test_a_source_whose_detail_went_missing_stays_active(store, monkeypatch):
    """
    Under-performing visibly rather than dropping a detail silently.
    The aggregate is written, it speaks for what it covers, and the
    entry holding `Ansible` and `Podman` keeps its place in the block.
    """
    _writes(
        monkeypatch,
        "Matériel : le NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM, "
        "SSD 256 Go",
    )

    report = _run(store)

    assert sorted(report[0]["held_back"]) == [2]
    assert "ansible" in report[0]["held_back"][2]
    contents = [e["content"] for e in rag.hot_entries(store)]
    assert NIPOGI[1] in contents
    assert len(contents) == 2


def test_an_aggregate_that_would_replace_one_entry_is_not_written(store, monkeypatch):
    """
    Quorum. An entry standing in for a single other entry is a rewrite
    of somebody's note, which is not what was asked for.
    """
    _writes(monkeypatch, "Le NiPoGi a 32 Go de RAM")

    report = _run(store)

    assert report[0]["refused"] == "quorum"
    assert len(rag.hot_entries(store)) == 4


def test_an_aggregate_no_shorter_than_its_sources_is_not_written(store, monkeypatch):
    """
    Budget. Making the block longer is the opposite of the job, and
    the hot tier's cap is a tripwire this pass is supposed to move
    away from, not toward.
    """
    # Padded with connectives, not with a second copy of the notes: a
    # repeated note trips the repetition gate first, and what is under
    # test here is length.
    padded = (
        "Matériel : le NiPoGi AM06PRO, Arch, processeur Ryzen 5500U, "
        "32 Go de RAM, SSD 256 Go, Ansible et services Podman, "
        + " ".join(["de le un avec et a"] * 6)
    )
    _writes(monkeypatch, padded)

    report = _run(store)

    assert report[0]["refused"] == "budget"
    assert len(rag.hot_entries(store)) == 4


def test_a_fold_inside_the_estimator_margin_is_refused():
    """
    Not a `>=`. The real store folded two entries of 27 estimated
    tokens into 24 while the same run logged estimate_drift at 21.4%:
    the gate was comparing two estimates whose error was seven times
    the gap it measured.
    """
    assert aggregate._BUDGET_MARGIN > 0.2


def test_the_pass_does_not_run_when_the_knob_is_off(store, monkeypatch):
    monkeypatch.setattr(aggregate, "COMPACTION_AGGREGATE", False)

    assert aggregate.maybe_aggregate() == []
    assert len(rag.hot_entries(store)) == 4


def test_a_store_that_cannot_be_written_is_not_an_error_the_user_reads(
    store, monkeypatch
):
    """
    The pass runs after a compaction that has already committed. It
    swallows everything for that reason, and the reason outlived the
    model call it was written for.
    """
    monkeypatch.setattr(aggregate, "COMPACTION_AGGREGATE", True)
    monkeypatch.setattr(aggregate.rag, "get_connection", lambda: 1 / 0)

    assert aggregate.maybe_aggregate() == []


# --- The fold that writes nothing ------------------------------------------


def test_a_note_is_absorbed_by_the_one_that_already_says_it(tmp_path, monkeypatch):
    """
    The real store's `steam` subject, which the budget gate refused as
    an aggregate and is right to: the line it would have written was
    #317 with a head glued on. Absorbing costs nothing, invents
    nothing, and leaves the user's own sentence in the block.
    """
    conn = _store(
        tmp_path / "steam.db",
        monkeypatch,
        STEAM,
    )

    report = _run(conn)

    assert report[0]["into"] == 2
    assert report[0]["folded"] == [1]
    assert [e["content"] for e in rag.hot_entries(conn)] == [STEAM[1]]
    conn.close()


def test_absorbing_writes_no_new_entry(tmp_path, monkeypatch):
    conn = _store(
        tmp_path / "steam.db",
        monkeypatch,
        STEAM,
    )
    before = conn.execute("SELECT count(*) FROM memory_entries").fetchone()[0]

    _run(conn)

    assert conn.execute("SELECT count(*) FROM memory_entries").fetchone()[0] == before
    conn.close()


def test_absorption_needs_no_quorum(tmp_path, monkeypatch):
    """
    Quorum refuses an entry that stands in for a single other entry,
    because that is a rewrite of somebody's note. Absorption rewrites
    nothing, so one source is enough -- and this is the case where the
    two rules differ, with min_sources at its default of two.
    """
    conn = _store(
        tmp_path / "steam.db",
        monkeypatch,
        STEAM,
    )

    report = _run(conn, min_sources=2)

    assert report[0].get("refused") is None
    assert len(report[0]["folded"]) == 1
    conn.close()


def test_an_entry_of_another_project_is_not_hidden_behind_one(tmp_path, monkeypatch):
    """
    A project is a namespace. An entry folded into an entry of another
    project drops out of the block under a name nobody filed it with.
    """
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "projects.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    rag.remember(conn, kind="fact", content=STEAM[0], project="jeux")
    rag.remember(conn, kind="fact", content=STEAM[1], project="forge")
    for content in FILLER:
        rag.remember(conn, kind="history_summary", content=content, project=None)

    report = _run(conn)

    assert report[0]["refused"] == "quorum"
    assert len(rag.hot_entries(conn)) == 2
    conn.close()
