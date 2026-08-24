"""
Tests for the shape of a fact the router writes.

The fixtures are real. Every string here was either stored in the live
store on 2026-08-24 or measured against it, which is the point: the
detector was not designed and then illustrated, it was read off three
entries the router had just written.
"""

import pytest

from forge.tools.memory import _is_telegraphic

# What the router wrote when asked to remember, 2026-08-24. #315 is
# the one that hurts: the sentence the user typed had "processeur" and
# "Ryzen" in it, and those are the two words that make #307 findable
# by a question about a processor.
TELEGRAMS = [
    "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible, services Podman",
    "Steam Deck, SteamOS, conteneurs Podman",
]

# Facts written as facts, from the same store.
SENTENCES = [
    "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go",
    "Le proxy podman écoute sur un socket unix",
    "Possède un Dell R710, configuration à préciser plus tard",
]


@pytest.mark.parametrize("text", TELEGRAMS)
def test_a_keyword_list_is_recognised(text):
    assert _is_telegraphic(text)


@pytest.mark.parametrize("text", SENTENCES)
def test_a_written_fact_is_not(text):
    assert not _is_telegraphic(text)


def test_a_short_enumeration_is_left_alone():
    """
    Two or three items is a sentence with commas in it, not a
    telegram. The floor exists so that ordinary punctuation does not
    trip the check.
    """
    assert not _is_telegraphic("Utilise podman, pas docker")


def test_nothing_is_ever_refused(monkeypatch):
    """
    The guard is advisory and stays advisory. A genuine enumeration --
    a list of service names, a list of ports -- trips it too, and
    losing a fact entirely is worse than storing a terse one. What it
    buys is that the degradation is VISIBLE when it happens instead of
    surfacing three weeks later as a question that cannot be answered.
    """
    from forge import rag
    from forge.tools import memory as memory_tool

    monkeypatch.setattr(rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(rag, "remember", lambda conn, kind, content, project: 42)
    monkeypatch.setattr(rag, "count_entries", lambda conn: {"by_kind": {"fact": 1}})
    monkeypatch.setattr(memory_tool, "_unfamiliar_words", lambda conn, text, eid: [])

    out = memory_tool._remember({"kind": "fact", "content": TELEGRAMS[0]})

    assert "[error]" not in out
    assert "mots-clés" in out


def test_a_written_fact_gets_no_note(monkeypatch):
    from forge import rag
    from forge.tools import memory as memory_tool

    monkeypatch.setattr(rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(rag, "remember", lambda conn, kind, content, project: 42)
    monkeypatch.setattr(rag, "count_entries", lambda conn: {"by_kind": {"fact": 1}})
    monkeypatch.setattr(memory_tool, "_unfamiliar_words", lambda conn, text, eid: [])

    out = memory_tool._remember({"kind": "fact", "content": SENTENCES[0]})

    assert "mots-clés" not in out


class _FakeConn:
    def close(self):
        pass
