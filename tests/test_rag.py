"""
Unit tests for forge.rag. The embedding server is monkeypatched at
rag._embed (same boundary the module itself calls through), so no
network / no real llama.cpp instance is needed.
"""

import pytest

from forge import rag

FAKE_DIM = 8  # small on purpose -- these tests don't care about EMBEDDING_DIM


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(
        rag, "_VEC_SCHEMA", rag._VEC_SCHEMA.replace("1024", str(FAKE_DIM))
    )
    connection = rag.get_connection()
    yield connection
    connection.close()


def _fake_embed(vec):
    def _inner(text):
        return vec

    return _inner


def test_remember_returns_incrementing_ids(conn, monkeypatch):
    monkeypatch.setattr(rag, "_embed", _fake_embed([0.1] * FAKE_DIM))

    first = rag.remember(
        conn, kind="decision", content="use sqlite-vec", project="forge"
    )
    second = rag.remember(conn, kind="todo", content="write api", project="forge")

    assert second == first + 1


def test_search_ranks_closer_vector_first(conn, monkeypatch):
    # Two near-identical vectors, one far away -- distance ordering
    # should put the near ones first, same shape as the real embedding
    # sanity check done manually against the live server.
    monkeypatch.setattr(
        rag, "_embed", _fake_embed([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    rag.remember(conn, kind="decision", content="close match", project="forge")

    monkeypatch.setattr(
        rag, "_embed", _fake_embed([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    )
    rag.remember(conn, kind="decision", content="far match", project="forge")

    monkeypatch.setattr(
        rag, "_embed", _fake_embed([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    results = rag.search(conn, query="anything", top_k=2)

    assert results[0]["content"] == "close match"
    assert results[1]["content"] == "far match"
    assert results[0]["distance"] < results[1]["distance"]


def test_search_filters_by_kind(conn, monkeypatch):
    monkeypatch.setattr(rag, "_embed", _fake_embed([0.5] * FAKE_DIM))
    rag.remember(conn, kind="decision", content="a decision", project="forge")
    rag.remember(conn, kind="todo", content="a todo", project="forge")

    results = rag.search(conn, query="anything", top_k=5, kind="todo")

    assert len(results) == 1
    assert results[0]["kind"] == "todo"


def test_search_filters_by_project(conn, monkeypatch):
    monkeypatch.setattr(rag, "_embed", _fake_embed([0.5] * FAKE_DIM))
    rag.remember(conn, kind="decision", content="forge thing", project="forge")
    rag.remember(conn, kind="decision", content="nipogi thing", project="nipogi")

    results = rag.search(conn, query="anything", top_k=5, project="nipogi")

    assert len(results) == 1
    assert results[0]["project"] == "nipogi"


def test_embedding_error_propagates(conn, monkeypatch):
    import requests

    def _raise(text):
        raise requests.ConnectionError("no server")

    monkeypatch.setattr(rag, "_embed", _raise)

    with pytest.raises(requests.ConnectionError):
        rag.remember(conn, kind="decision", content="whatever", project=None)


def test_real_embed_wraps_request_failure_in_embedding_error(monkeypatch):
    """Unlike the tests above (which bypass _embed), this checks _embed
    itself: a real request failure must surface as rag.EmbeddingError,
    since that's what api.py catches to return a 502."""
    import requests

    def _raise_connection_error(*a, **kw):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(rag.requests, "post", _raise_connection_error)

    with pytest.raises(rag.EmbeddingError):
        rag._embed("hello")
