"""
Tests for splitting long text before embedding it.

Regression cover for a real failure: /embedding answered 400 Bad
Request for compaction while short recall queries kept working. The
whole evicted history block was posted in one request, past the
embedding server's physical batch size. Compaction failed on every
turn, memory fell back to the drop-oldest hard cap, and KV-cache
reuse went with it.
"""

import math

import pytest

import forge.rag as rag_mod


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setattr(rag_mod, "EMBEDDING_MAX_CHARS", 20)
    monkeypatch.setattr(rag_mod, "EMBEDDING_MAX_CHUNKS", 16)


# ── _split_for_embedding ─────────────────────────────────────────────


def test_short_text_stays_a_single_chunk(small_chunks):
    assert rag_mod._split_for_embedding("court") == ["court"]


def test_every_chunk_respects_the_limit(small_chunks):
    text = "\n".join(f"role: message numéro {i}" for i in range(30))
    chunks = rag_mod._split_for_embedding(text)

    assert len(chunks) > 1
    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks) == text  # nothing lost, nothing duplicated


def test_a_single_oversized_line_is_hard_split(small_chunks):
    text = "x" * 55
    chunks = rag_mod._split_for_embedding(text)

    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks) == text


def test_empty_text_still_yields_one_chunk(small_chunks):
    assert rag_mod._split_for_embedding("") == [""]


# ── _embed ───────────────────────────────────────────────────────────


def test_short_input_is_embedded_in_one_request(small_chunks, monkeypatch):
    calls = []
    monkeypatch.setattr(rag_mod, "_embed_one", lambda t: calls.append(t) or [1.0, 0.0])

    assert rag_mod._embed("court") == [1.0, 0.0]
    assert calls == ["court"]


def test_long_input_is_chunked_instead_of_posted_whole(small_chunks, monkeypatch):
    sent = []

    def fake(text):
        sent.append(text)
        return [1.0, 0.0]

    monkeypatch.setattr(rag_mod, "_embed_one", fake)
    rag_mod._embed("\n".join(f"role: message {i}" for i in range(20)))

    assert len(sent) > 1
    assert all(len(t) <= 20 for t in sent)


def test_chunked_result_is_a_unit_vector(small_chunks, monkeypatch):
    vectors = iter([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(rag_mod, "_embed_one", lambda t: next(vectors))

    result = rag_mod._embed("a" * 20 + "b" * 20)  # exactly two chunks

    assert math.isclose(math.sqrt(sum(v * v for v in result)), 1.0)
    assert math.isclose(result[0], result[1])


def test_chunk_count_is_capped(small_chunks, monkeypatch):
    monkeypatch.setattr(rag_mod, "EMBEDDING_MAX_CHUNKS", 3)
    sent = []
    monkeypatch.setattr(rag_mod, "_embed_one", lambda t: sent.append(t) or [1.0, 0.0])

    rag_mod._embed("\n".join(f"role: message {i}" for i in range(40)))

    assert len(sent) == 3


def test_a_failing_chunk_still_raises_embedding_error(small_chunks, monkeypatch):
    def fake(text):
        raise rag_mod.EmbeddingError("400 Client Error: Bad Request")

    monkeypatch.setattr(rag_mod, "_embed_one", fake)

    with pytest.raises(rag_mod.EmbeddingError):
        rag_mod._embed("\n".join(f"role: message {i}" for i in range(20)))
