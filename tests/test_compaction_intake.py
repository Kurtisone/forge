"""
What compaction puts INTO the vector store, as opposed to what it
takes out of the history.

Everything here comes from reading the real store on 2026-08-22, the
first day anything could enumerate it: 16 entries, 11 of them whole
compacted blocks, one holding raw router JSON, and a measured 0.27 of
distance lost to burying a sentence in a block.
"""

import pytest

from forge import compaction, rag, transcript


@pytest.fixture
def indexed(monkeypatch):
    """Capture what reaches the store instead of embedding it."""
    captured: list[str] = []

    class _FakeConn:
        def close(self):
            pass

    def _remember_many(conn, kind, contents, project):
        captured.extend(contents)
        return list(range(10, 10 + len(contents)))

    monkeypatch.setattr(compaction.rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(compaction.rag, "remember_many", _remember_many)
    return captured


def _m(role: str, content: str, mid: int = 0) -> dict:
    return {"id": mid, "role": role, "content": content, "pinned": False}


def test_each_exchange_becomes_its_own_entry(indexed):
    compaction._strategy_rag_pointer(
        [
            _m("user", "quel port ?", 1),
            _m("assistant", "8080", 2),
            _m("user", "et l'hôte ?", 3),
            _m("assistant", "localhost", 4),
        ]
    )

    assert indexed == [
        "user: quel port ?\nassistant: 8080",
        "user: et l'hôte ?\nassistant: localhost",
    ]


def test_the_pointer_names_the_range_it_created(indexed):
    summary = compaction._strategy_rag_pointer(
        [
            _m("user", "a", 1),
            _m("assistant", "b", 2),
            _m("user", "c", 3),
            _m("assistant", "d", 4),
        ]
    )

    assert "#10-#11 (2 entrées)" in summary["content"]
    assert "4 messages" in summary["content"]


def test_a_single_entry_is_named_without_a_range(indexed):
    summary = compaction._strategy_rag_pointer(
        [_m("user", "une question", 1), _m("assistant", "une réponse", 2)]
    )

    assert "#10," in summary["content"]
    assert "-#" not in summary["content"]


def test_the_pointer_is_recognisable_by_the_regex_that_looks_for_it():
    """
    The builder and the detector are two statements of the same string
    and would drift apart silently -- a pointer that stops matching is
    a pointer that gets indexed as if it were conversation, and
    nothing fails.
    """
    for ids in ([], [7], [7, 8, 9]):
        assert transcript.POINTER_RE.match(transcript.pointer(12, ids))


def test_an_earlier_pointer_is_not_indexed_again(indexed):
    """
    A pointer is a reference to another entry. Stored, it becomes a
    memory whose whole content is "N messages were compacted, see
    #12": it answers no question and sits at middling distance from
    every one of them.
    """
    compaction._strategy_rag_pointer(
        [
            _m("system", transcript.pointer(59, [12]), 1),
            _m("user", "une question", 2),
            _m("assistant", "une réponse", 3),
        ]
    )

    assert indexed == ["user: une question\nassistant: une réponse"]


def test_a_router_json_envelope_is_unwrapped_not_stored_raw(indexed):
    """
    Entry #9 of the real store was raw router JSON, swallowed from an
    assistant turn by an older version. The envelope is the noise; the
    answer inside it is real.
    """
    compaction._strategy_rag_pointer(
        [
            _m("user", "explique le cache KV", 1),
            _m(
                "assistant",
                '{"tool": "chat", "content": "Le cache KV garde les clés et '
                'valeurs déjà calculées pour ne pas refaire le prefill."}',
                2,
            ),
        ]
    )

    assert len(indexed) == 1
    assert indexed[0].startswith("user: explique le cache KV\nassistant: Le cache KV")
    assert '"tool"' not in indexed[0]


def test_empty_messages_are_not_indexed(indexed):
    compaction._strategy_rag_pointer(
        [_m("user", "   ", 1), _m("assistant", "une vraie réponse", 2)]
    )

    assert indexed == ["assistant: une vraie réponse"]


def test_a_block_with_nothing_indexable_says_so(indexed):
    summary = compaction._strategy_rag_pointer(
        [_m("system", transcript.pointer(59, [12]), 1)]
    )

    assert indexed == []
    assert "aucune entrée mémoire créée" in summary["content"]
    assert "#" not in summary["content"].split("--")[1]


def test_the_embedding_server_being_down_still_stops_compaction(indexed, monkeypatch):
    def _boom(conn, kind, contents, project):
        raise rag.EmbeddingError("down")

    monkeypatch.setattr(compaction.rag, "remember_many", _boom)

    with pytest.raises(compaction.CompactionError):
        compaction._strategy_rag_pointer([_m("user", "une question", 1)])
