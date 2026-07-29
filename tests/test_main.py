"""
Tests for the !remember / !recall REPL commands added to forge.main
in v3.7, and for the command dispatch fix (only the keyword is
lowercased, never the user's actual content).

rag._embed is monkeypatched, same boundary used in test_rag.py and
test_api.py -- no network / no real embedding server involved.
"""

import pytest

import forge.main as main_mod
from forge import rag


@pytest.fixture
def _rag_conn(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)


# ── !remember ─────────────────────────────────────────────────────────


def test_remember_stores_entry_with_project(_rag_conn, capsys):
    main_mod._handle_command("!remember decision forge Utiliser SQLite-vec pour le RAG")
    out = capsys.readouterr().out
    assert "remembered #1" in out


def test_remember_accepts_dash_as_no_project(_rag_conn, capsys):
    main_mod._handle_command("!remember todo - Terminer la doc")
    out = capsys.readouterr().out
    assert "remembered #1" in out


def test_remember_preserves_content_case(_rag_conn, monkeypatch):
    captured = {}
    original_remember = rag.remember

    def _spy(conn, kind, content, project):
        captured["content"] = content
        return original_remember(conn, kind=kind, content=content, project=project)

    monkeypatch.setattr(rag, "remember", _spy)
    main_mod._handle_command("!remember decision forge Utiliser SQLite-vec")

    assert captured["content"] == "Utiliser SQLite-vec"


def test_remember_rejects_invalid_kind(_rag_conn, capsys):
    main_mod._handle_command("!remember note forge something")
    out = capsys.readouterr().out
    assert "kind must be" in out


def test_remember_shows_usage_when_missing_args(_rag_conn, capsys):
    main_mod._handle_command("!remember decision")
    out = capsys.readouterr().out
    assert "Usage: !remember" in out


def test_remember_reports_embedding_failure(_rag_conn, monkeypatch, capsys):
    def _raise(text):
        raise rag.EmbeddingError("connection refused")

    monkeypatch.setattr(rag, "_embed", _raise)
    main_mod._handle_command("!remember decision forge some content")
    out = capsys.readouterr().out
    assert "remember failed" in out


# ── !recall ───────────────────────────────────────────────────────────


def test_recall_finds_stored_entry(_rag_conn, capsys):
    main_mod._handle_command("!remember decision forge Utiliser SQLite-vec pour le RAG")
    capsys.readouterr()  # discard the !remember output

    main_mod._handle_command("!recall sqlite")
    out = capsys.readouterr().out
    assert "Utiliser SQLite-vec pour le RAG" in out
    assert "decision/forge" in out


def test_recall_reports_no_matches_on_empty_db(_rag_conn, capsys):
    main_mod._handle_command("!recall anything")
    out = capsys.readouterr().out
    assert "no matches" in out


def test_recall_shows_usage_when_query_empty(_rag_conn, capsys):
    main_mod._handle_command("!recall   ")
    out = capsys.readouterr().out
    assert "Usage: !recall" in out


def test_recall_reports_embedding_failure(_rag_conn, monkeypatch, capsys):
    def _raise(text):
        raise rag.EmbeddingError("connection refused")

    monkeypatch.setattr(rag, "_embed", _raise)
    main_mod._handle_command("!recall anything")
    out = capsys.readouterr().out
    assert "recall failed" in out


# ── dispatch ──────────────────────────────────────────────────────────


def test_unknown_command_prints_hint(capsys):
    main_mod._handle_command("!bogus")
    out = capsys.readouterr().out
    assert "unknown command" in out
    assert "!help" in out


def test_command_keyword_is_case_insensitive(_rag_conn, capsys):
    main_mod._handle_command("!REMEMBER decision forge Some Content")
    out = capsys.readouterr().out
    assert "remembered #1" in out
