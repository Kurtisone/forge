"""
Tests for deploy/rag_resplit.py, the one-shot migration that re-slices
compaction blocks already in the store.

Imported by file path since deploy/ is an ops script, not part of the
installable forge package -- same as tests/test_podman_ro_proxy.py.

What matters here is that a dry run writes nothing (it is the default,
so it is the mode that runs by accident) and that a real run does not
lose an entry it could not improve.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from forge import rag

_SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "rag_resplit.py"
_spec = importlib.util.spec_from_file_location("rag_resplit", _SCRIPT)
rag_resplit = importlib.util.module_from_spec(_spec)
sys.modules["rag_resplit"] = rag_resplit
_spec.loader.exec_module(rag_resplit)


_BLOCK = (
    "user: quel port utilise le serveur de test ?\n"
    "assistant: le 8080\n"
    "user: et sur quelle machine tourne Forge ?\n"
    "assistant: sur le Steam Deck pour l'instant"
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    db = tmp_path / "rag.db"
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(db))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    rag.remember(conn, kind="history_summary", content=_BLOCK, project=None)
    rag.remember(conn, kind="fact", content="Le NiPoGi a 32 Go de RAM", project=None)
    rag.remember(
        conn,
        kind="history_summary",
        content="user: une seule question\nassistant: une seule réponse",
        project=None,
    )
    conn.close()
    return db


def _invoke(db, *flags):
    argv = ["rag_resplit.py", "--db", str(db), *flags]
    saved, sys.argv = sys.argv, argv
    try:
        return rag_resplit.main()
    finally:
        sys.argv = saved


def test_dry_run_writes_nothing(store, capsys):
    assert _invoke(store) == 0

    conn = rag.get_connection()
    try:
        assert rag.count_entries(conn)["by_kind"] == {"history_summary": 2, "fact": 1}
    finally:
        conn.close()
    assert "dry run" in capsys.readouterr().out


def test_apply_replaces_a_block_with_its_exchanges(store):
    assert _invoke(store, "--apply") == 0

    conn = rag.get_connection()
    try:
        entries = rag.list_entries(conn, kind="history_summary")
        contents = sorted(e["content"] for e in entries)
    finally:
        conn.close()

    assert contents == sorted(
        [
            "user: quel port utilise le serveur de test ?\nassistant: le 8080",
            (
                "user: et sur quelle machine tourne Forge ?\n"
                "assistant: sur le Steam Deck pour l'instant"
            ),
            "user: une seule question\nassistant: une seule réponse",
        ]
    )


def test_an_entry_that_is_already_one_exchange_keeps_its_id(store):
    conn = rag.get_connection()
    try:
        before = {
            e["content"]: e["id"]
            for e in rag.list_entries(conn, kind="history_summary")
        }
    finally:
        conn.close()
    single = "user: une seule question\nassistant: une seule réponse"

    _invoke(store, "--apply")

    conn = rag.get_connection()
    try:
        after = {
            e["content"]: e["id"]
            for e in rag.list_entries(conn, kind="history_summary")
        }
    finally:
        conn.close()

    # Rewriting it would spend an embedding call to produce the same
    # row under a new id.
    assert after[single] == before[single]


def test_other_kinds_are_left_alone(store):
    _invoke(store, "--apply")

    conn = rag.get_connection()
    try:
        facts = rag.list_entries(conn, kind="fact")
    finally:
        conn.close()

    assert [f["content"] for f in facts] == ["Le NiPoGi a 32 Go de RAM"]


def test_the_vectors_follow_the_entries(store):
    """
    forget() deletes from both tables; remember_many writes to both. A
    migration that lost the correspondence would leave vectors search
    can still match and list_entries can no longer show.
    """
    _invoke(store, "--apply")

    conn = rag.get_connection()
    try:
        entry_ids = {e["id"] for e in rag.list_entries(conn)}
        vector_ids = {r[0] for r in conn.execute("SELECT rowid FROM memory_vectors")}
    finally:
        conn.close()

    assert entry_ids == vector_ids


def test_a_missing_database_is_reported_not_created(tmp_path, capsys):
    missing = tmp_path / "nope.db"

    assert _invoke(missing) == 1
    assert not missing.exists()


def test_backup_is_written_before_applying(store, tmp_path):
    backup = tmp_path / "backup.db"

    _invoke(store, "--apply", "--backup", str(backup))

    assert backup.exists()


def test_an_entry_that_is_only_a_pointer_is_reported_not_deleted(
    tmp_path, monkeypatch, capsys
):
    """
    The migration of 2026-08-22 turned pointers into entries because
    this script cut the same way compaction did but skipped its
    filtering. It filters now -- and an entry left with nothing at all
    is reported rather than removed: deleting a row nobody asked to
    delete is not a migration's job.
    """
    from forge import transcript

    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    rag.remember(
        conn,
        kind="history_summary",
        content=f"system: {transcript.pointer(59, [12])}",
        project=None,
    )
    conn.close()

    _invoke(tmp_path / "rag.db", "--apply")

    conn = rag.get_connection()
    try:
        assert rag.count_entries(conn)["total"] == 1
    finally:
        conn.close()
    assert "inert" in capsys.readouterr().out


def test_a_pointer_inside_a_block_does_not_become_an_entry(tmp_path, monkeypatch):
    from forge import transcript

    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    rag.remember(
        conn,
        kind="history_summary",
        content=(
            f"system: {transcript.pointer(59, [12])}\n"
            "user: une question\nassistant: une réponse\n"
            "user: une autre\nassistant: une autre réponse"
        ),
        project=None,
    )
    conn.close()

    _invoke(tmp_path / "rag.db", "--apply")

    conn = rag.get_connection()
    try:
        contents = [e["content"] for e in rag.list_entries(conn)]
    finally:
        conn.close()

    assert contents == [
        "user: une autre\nassistant: une autre réponse",
        "user: une question\nassistant: une réponse",
    ]
