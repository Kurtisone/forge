"""
The hybrid harness has to RUN before it can measure anything.

Same reasoning as tests/test_recall_expansion_bench.py: a harness that
crashes two hundred lines in costs a real round trip to the Deck to
find out. The embedder is a stub, so the distances here mean nothing
and asserting on them would be the same mistake pointed the other
way. What is asserted is that every path is reachable -- the rescue,
the intruder, the wrong entry -- and that the refusals still refuse.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from forge import rag

_SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "rag_hybrid.py"
sys.path.insert(0, str(_SCRIPT.parent))
_spec = importlib.util.spec_from_file_location("rag_hybrid", _SCRIPT)
rag_hybrid = importlib.util.module_from_spec(_spec)
sys.modules["rag_hybrid"] = rag_hybrid
_spec.loader.exec_module(rag_hybrid)

FAKE_DIM = 8
NEAR = [1.0] + [0.0] * (FAKE_DIM - 1)
MIDDLING = [0.7071, 0.7071] + [0.0] * (FAKE_DIM - 2)
FAR = [0.0, 1.0] + [0.0] * (FAKE_DIM - 2)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """
    A miniature of the real store: one telegraphic fact nothing can
    reach by meaning, one archived refusal that quotes the question it
    failed to answer, and enough filler for a frequency rule to have
    something to measure.
    """
    path = tmp_path / "bench.db"
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(path))
    monkeypatch.setattr(
        rag, "_VEC_SCHEMA", rag._VEC_SCHEMA.replace("1024", str(FAKE_DIM))
    )
    conn = rag.get_connection()

    monkeypatch.setattr(rag, "_embed", lambda text: MIDDLING)
    for i in range(20):
        rag.remember(
            conn, kind="fact", content=f"Note anodine {i} sans rapport", project=None
        )

    monkeypatch.setattr(rag, "_embed", lambda text: FAR)
    telegram = rag.remember(
        conn,
        kind="fact",
        content="Steam Deck, SteamOS, conteneurs Podman",
        project=None,
    )
    rag.remember(
        conn,
        kind=rag.ARCHIVED_KIND,
        content="user: Comment s'appelle mon chat ?\nassistant: Je ne sais pas encore",
        project=None,
    )
    conn.close()

    # The query side, from here on.
    monkeypatch.setattr(rag, "_embed", lambda text: NEAR)
    yield str(path), telegram


def _run(monkeypatch, *argv) -> int:
    monkeypatch.setattr(sys, "argv", ["rag_hybrid.py", *argv])
    return rag_hybrid.main()


def test_a_placeholder_question_is_refused(monkeypatch, store):
    db, _ = store

    assert _run(monkeypatch, "--db", db, "--hit", "<la question du 22/08>") == 1


def test_expect_has_to_line_up_with_hit(monkeypatch, store):
    db, _ = store

    assert (
        _run(
            monkeypatch,
            "--db",
            db,
            "--hit",
            "une question",
            "--expect",
            "1",
            "--expect",
            "2",
        )
        == 1
    )


def test_a_missing_database_is_refused(monkeypatch, tmp_path):
    assert _run(monkeypatch, "--db", str(tmp_path / "nope.db"), "--hit", "q") == 1


def test_the_word_channel_reaches_what_the_vector_channel_cannot(
    monkeypatch, store, capsys
):
    db, telegram = store

    assert (
        _run(
            monkeypatch,
            "--db",
            db,
            "--cutoff",
            "0.88",
            "--hit",
            "Sur quoi tournent mes conteneurs Podman ?",
            "--expect",
            str(telegram),
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "REACHED      1" in out


def test_a_refused_question_answered_by_words_is_counted_as_an_intruder(
    monkeypatch, store, capsys
):
    """
    The cost side. An archived refusal quotes the question it failed to
    answer, which makes it an excellent word match for it -- and a
    harness that only counted wins would print none of this.
    """
    db, _ = store

    assert _run(monkeypatch, "--db", db, "--miss", "Comment s'appelle mon chat ?") == 0

    out = capsys.readouterr().out
    assert "INTRUDERS    1" in out
    assert "answered a MISS with" in out


def test_no_vector_does_not_claim_a_rescue_it_never_measured(
    monkeypatch, store, capsys
):
    db, telegram = store

    assert (
        _run(
            monkeypatch,
            "--db",
            db,
            "--no-vector",
            "--hit",
            "Sur quoi tournent mes conteneurs Podman ?",
            "--expect",
            str(telegram),
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "REACHED      0" in out
    assert "--no-vector" in out


def test_several_max_df_values_print_several_columns(monkeypatch, store, capsys):
    db, telegram = store

    _run(
        monkeypatch,
        "--db",
        db,
        "--no-vector",
        "--max-df",
        "0.05",
        "--max-df",
        "0.5",
        "--hit",
        "Sur quoi tournent mes conteneurs Podman ?",
        "--expect",
        str(telegram),
    )

    out = capsys.readouterr().out
    assert "word 0.05" in out
    assert "word 0.5" in out


def test_an_entry_inside_the_top_k_but_beyond_the_cutoff_is_not_delivered():
    """
    The distinction the win column rests on: coming back is not the
    same as reaching synthesis.
    """
    rows = [{"id": 7, "distance": 1.4}]

    assert rag_hybrid._delivered(rows, "7", 0.88) is False
    assert rag_hybrid._delivered(rows, "7", None) is True
    assert rag_hybrid._delivered(rows, "9", 0.88) is False
