"""
The query instruction, and the asymmetry that makes it free.

The one property worth a test: a document must never be embedded with
a question's instruction glued to its front. If that ever happened the
store would hold two incompatible kinds of vector, every distance
measured before the change would become meaningless, and nothing would
report it -- retrieval would just quietly get worse.
"""

import pytest

from forge import rag


@pytest.fixture
def embedded(monkeypatch):
    """Capture exactly what text reaches the embedding server."""
    seen: list[str] = []

    def _embed(text: str):
        seen.append(text)
        return [0.0] * 1024

    monkeypatch.setattr(rag, "_embed", _embed)
    return seen


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "t.db"))
    c = rag.get_connection()
    yield c
    c.close()


def test_a_query_is_wrapped(embedded, conn):
    rag.search(conn, query="Quel processeur a mon NiPoGi ?")

    assert len(embedded) == 1
    assert embedded[0].startswith("Instruct: ")
    assert embedded[0].endswith("\nQuery: Quel processeur a mon NiPoGi ?")


def test_a_document_is_not(embedded, conn):
    rag.remember(conn, kind="fact", content="NiPoGi AM06PRO, Ryzen 5500U", project=None)

    assert embedded == ["NiPoGi AM06PRO, Ryzen 5500U"]


def test_a_document_that_looks_like_a_question_is_still_not_wrapped(embedded, conn):
    # One entry per exchange means stored text routinely IS a question.
    # The wrapper is decided by the call path, never by the text.
    rag.remember(
        conn,
        kind="history_summary",
        content="user: Quel processeur a mon NiPoGi ?\nassistant: Un Ryzen 5500U.",
        project=None,
    )

    assert "Instruct:" not in embedded[0]


def test_empty_instruction_disables_it(embedded, conn, monkeypatch):
    monkeypatch.setattr(rag, "EMBEDDING_QUERY_INSTRUCT", "  ")

    rag.search(conn, query="Quel processeur a mon NiPoGi ?")

    assert embedded == ["Quel processeur a mon NiPoGi ?"]
