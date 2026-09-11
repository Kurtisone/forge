"""
Tests for the pass itself: what it writes, and the four ways it
refuses to.

The rule the refusals share is the one this branch was opened on:
NOTHING IS WRITTEN UNLESS IT IS GOING TO REPLACE SOMETHING. Every gate
is a comparison between texts, so all of them run before the entry is
stored -- an aggregate that would be one more overlapping line in the
block instead of one fewer never reaches the store at all.

The model is stubbed. What a 9B actually writes under the grammar is a
question for bench/rag_aggregate.py against a copy of the real store;
what is pinned here is what the pass does with an answer once it has
one.
"""

import json

import pytest

from forge import aggregate, rag

NIPOGI = [
    "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go",
    "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible, services Podman",
    "Le NiPoGi a 32 Go de RAM",
]

#: Every word here is either a source word or a connective of the
#: corpus below. That is not stylistic care, it is the closure gate:
#: the first draft of this fixture said "tourne sous Arch" and was
#: refused, because nobody wrote `tourne` or `sous`.
GOOD = (
    "Matériel : le NiPoGi AM06PRO, Arch, processeur Ryzen 5500U, "
    "32 Go de RAM, SSD 256 Go, Ansible et services Podman"
)

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


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    for content in NIPOGI:
        rag.remember(conn, kind="fact", content=content, project=None)
    for content in FILLER:
        rag.remember(conn, kind="history_summary", content=content, project=None)
    yield conn
    conn.close()


def _answers(monkeypatch, text):
    calls = []

    def fake(prompt, grammar=None):
        calls.append((prompt, grammar))
        return text

    monkeypatch.setattr(aggregate, "call_llm", fake)
    return calls


def _run(conn, min_sources=2):
    return aggregate.run_pass(conn, max_df=0.2, min_sources=min_sources)


def test_the_block_goes_from_three_lines_to_one(store, monkeypatch):
    _answers(monkeypatch, GOOD)

    report = _run(store)

    assert len(report) == 1
    assert len(report[0]["folded"]) == 3
    assert [e["content"] for e in rag.hot_entries(store)] == [GOOD]


def test_the_sources_are_still_there_and_still_findable(store, monkeypatch):
    _answers(monkeypatch, GOOD)
    _run(store)

    stored = {r[0] for r in store.execute("SELECT content FROM memory_entries")}
    assert set(NIPOGI) <= stored


def test_the_call_is_constrained_by_the_subject_lexicon(store, monkeypatch):
    calls = _answers(monkeypatch, GOOD)
    _run(store)

    _, grammar = calls[0]
    assert '"NiPoGi"' in grammar
    assert '"Nvidia"' not in grammar


def test_a_word_from_nowhere_is_not_stored(store, monkeypatch):
    """
    The gate that separates a fact aggregated from a fact invented.
    Reached only on a provider with no grammar -- llama.cpp cannot
    sample this sentence at all -- which is exactly why it exists.
    """
    _answers(monkeypatch, "Le NiPoGi AM06PRO a 32 Go de RAM et une carte Nvidia")

    report = _run(store)

    assert report[0]["refused"] == "closure"
    assert "nvidia" in report[0]["invented"]
    assert len(rag.hot_entries(store)) == 3


def test_a_source_whose_detail_went_missing_stays_active(store, monkeypatch):
    """
    Under-performing visibly rather than dropping a detail silently.
    The aggregate is written, it speaks for what it covers, and the
    entry holding `Ansible` and `Podman` keeps its place in the block.
    """
    _answers(
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
    _answers(monkeypatch, "Le NiPoGi a 32 Go de RAM")

    report = _run(store)

    assert report[0]["refused"] == "quorum"
    assert len(rag.hot_entries(store)) == 3


def test_an_aggregate_no_shorter_than_its_sources_is_not_written(store, monkeypatch):
    """
    Budget. Making the block longer is the opposite of the job, and
    the hot tier's cap is a tripwire this pass is supposed to move
    away from, not toward.
    """
    _answers(monkeypatch, " ".join(NIPOGI) + " " + " ".join(NIPOGI))

    report = _run(store)

    assert report[0]["refused"] == "budget"
    assert len(rag.hot_entries(store)) == 3


def test_a_provider_failure_changes_nothing(store, monkeypatch):
    from forge.errors import ProviderError

    def boom(prompt, grammar=None):
        raise ProviderError("llama-server is down")

    monkeypatch.setattr(aggregate, "call_llm", boom)

    report = _run(store)

    assert report[0]["refused"] == "provider"
    assert len(rag.hot_entries(store)) == 3


def test_a_routing_decision_is_unwrapped_before_it_becomes_a_fact(store, monkeypatch):
    """
    The failure compaction's llm_summary strategy has a paragraph
    about, arriving one module over: with no grammar the model can
    answer with the router's JSON, and here that envelope would be
    stored as a fact about the user.
    """
    _answers(
        monkeypatch,
        json.dumps({"tool": "chat", "content": GOOD, "done": True}, ensure_ascii=False),
    )

    report = _run(store)

    assert report[0].get("id")
    assert rag.hot_entries(store)[0]["content"] == GOOD


def test_the_pass_does_not_run_when_the_knob_is_off(store, monkeypatch):
    calls = _answers(monkeypatch, GOOD)
    monkeypatch.setattr(aggregate, "COMPACTION_AGGREGATE", False)

    assert aggregate.maybe_aggregate() == []
    assert calls == []
