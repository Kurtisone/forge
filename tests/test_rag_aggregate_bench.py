"""
The aggregation harness has to RUN before it can measure anything.

Same reason as tests/test_rag_hot_tier_bench.py: a harness is meant to
be copied into the container and run once against a copy of the real
store, and a crash two hundred lines in wastes a round trip on the
Deck.

One assertion here is not about plumbing. This harness is the first in
bench/ that can WRITE, and the property that makes it safe to point at
a path is that it writes only under --apply. A regression there would
be discovered by finding aggregates in a store nobody meant to change.

That property got easier to break, not harder, when the model left:
the dry run now calls the real pass rather than stopping before it,
and what keeps it a dry run is one argument threaded through to the
one place that stores anything.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from forge import rag

_SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "rag_aggregate.py"
_spec = importlib.util.spec_from_file_location("rag_aggregate", _SCRIPT)
rag_aggregate = importlib.util.module_from_spec(_spec)
sys.modules["rag_aggregate"] = rag_aggregate
sys.path.insert(0, str(_SCRIPT.parent))
_spec.loader.exec_module(rag_aggregate)


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "rag.db"
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(path))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    for content in (
        "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM",
        "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, Ansible",
        "Le NiPoGi a 32 Go de RAM",
        "Possède un Steam Deck sous SteamOS",
    ):
        rag.remember(conn, kind="fact", content=content, project=None)
    conn.close()
    return str(path)


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["rag_aggregate", *argv])
    return rag_aggregate.main()


def test_it_runs_and_prints_the_groups_and_the_block(db, monkeypatch, capsys):
    assert _run(monkeypatch, ["--db", db, "--max-df", "0.5"]) == 0

    out = capsys.readouterr().out
    assert "BLOCK BEFORE" in out
    assert "SUBJECT" in out


def test_the_dry_run_shows_the_line_and_what_it_would_fold(db, monkeypatch, capsys):
    """
    The half that used to cost a model call per subject. A dry run
    that shows only the groups leaves the thing worth reading -- the
    text about to stand in for somebody's notes -- behind a flag that
    writes.
    """
    assert _run(monkeypatch, ["--db", db, "--max-df", "0.5"]) == 0

    out = capsys.readouterr().out
    assert "WOULD FOLD" in out
    assert "BLOCK WOULD BE" in out


def test_it_writes_and_folds_under_apply(db, monkeypatch, capsys):
    assert _run(monkeypatch, ["--db", db, "--max-df", "0.5", "--apply"]) == 0

    assert "FOLDED" in capsys.readouterr().out
    conn = rag.get_connection()
    try:
        assert (
            conn.execute(
                "SELECT count(*) FROM memory_entries WHERE superseded_by IS NOT NULL"
            ).fetchone()[0]
            > 0
        )
    finally:
        conn.close()


def test_it_writes_nothing_without_apply(db, monkeypatch):
    _run(monkeypatch, ["--db", db, "--max-df", "0.5"])

    conn = rag.get_connection()
    try:
        assert len(rag.hot_entries(conn)) == 4
        assert (
            conn.execute(
                "SELECT count(*) FROM memory_entries WHERE superseded_by IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_a_store_with_nothing_to_aggregate_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "empty.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    rag.get_connection().close()

    assert _run(monkeypatch, ["--db", str(tmp_path / "empty.db")]) == 1
    assert "nothing to aggregate" in capsys.readouterr().out


def test_a_missing_database_is_named_rather_than_traced(monkeypatch, capsys):
    assert _run(monkeypatch, ["--db", "/tmp/does-not-exist.db"]) == 1
    assert "does not exist" in capsys.readouterr().out
