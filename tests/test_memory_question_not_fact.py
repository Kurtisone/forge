"""
A question the user asked is never a fact the user stated.

THE FIXTURE IS REAL, and it is the whole reason this file exists.
From traces.jsonl, 2026-09-11:

    user: Je possède un serveur ?
    payload: {"action":"remember","kind":"fact","content":"Possède un serveur"}

The question mark is the only thing that separated that turn from a
statement, and it is gone from the payload -- so nothing downstream of
the router can tell what happened. Asked the same question a week
later, the store answers it with itself.

The router prompt already says to use the memory tool "only when the
user explicitly asks you to remember/save something". This is what
replaces that sentence with something the model cannot break.

WHY THE RULE COSTS NOTHING HERE, measured over every memory routing in
the trace file: 19 remembers, 18 of them declarative and legitimate,
one of them a question, and it is the bug above. No genuine remember
has ever been phrased as a question on this store.
"""

import json

import pytest

from forge import rag, turn
from forge.tools import memory as memory_tool


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    yield
    turn.clear()


def _remember(text: str) -> str:
    return memory_tool.run(
        json.dumps({"action": "remember", "kind": "fact", "content": text})
    )


def _stored() -> list[str]:
    conn = rag.get_connection()
    try:
        return [e["content"] for e in rag.list_entries(conn, limit=50)]
    finally:
        conn.close()


def test_the_turn_from_the_traces_is_not_written():
    turn.set_input("Je possède un serveur ?")

    _remember("Possède un serveur")

    assert _stored() == []


def test_a_statement_is_still_written():
    turn.set_input("Je possède un serveur")

    _remember("Possède un serveur")

    assert _stored() == ["Possède un serveur"]


def test_the_question_is_answered_rather_than_dropped():
    """
    "Je possède un serveur ?" is a lookup. The other 39 memory
    routings in the same trace file are recalls of questions exactly
    like it, so the honest thing to do with this one is the same.
    """
    turn.set_input("Je possède un serveur ?")

    answer = _remember("Possède un serveur")

    assert answer
    assert "[error]" not in answer


def test_a_long_message_ending_in_a_question_mark_still_writes():
    """
    A message with a question mark at the end is not a question. Same
    bound delegation.py has always used, now in one place.
    """
    turn.set_input(
        "Retiens que mon NiPoGi AM06PRO tourne sous Arch avec 32 Go de RAM, tu peux ?"
    )

    _remember("Le NiPoGi AM06PRO tourne sous Arch avec 32 Go de RAM")

    assert _stored() == ["Le NiPoGi AM06PRO tourne sous Arch avec 32 Go de RAM"]


def test_a_write_outside_a_run_is_untouched():
    """
    !remember from the REPL, /remember over HTTP, a test calling the
    tool directly: no turn was set, so there is no shape to read and
    the explicit ask is the human's.
    """
    turn.clear()

    _remember("Le serveur de test tourne sur le port 8080")

    assert _stored() == ["Le serveur de test tourne sur le port 8080"]
