"""
The hot-tier harness has to RUN before it can measure anything.

Same reason as tests/test_rag_dilution_bench.py: a harness is meant to
be copied into the container and run once against the real store, and
a crash two hundred lines in wastes a round trip on the Deck. Five
faults reached the Deck in one session before bench/ had any coverage
at all.

The numbers here are meaningless -- the store is four fixtures -- and
asserting on them would be the same mistake in another direction.
What is asserted is that the plumbing runs, that SUBSUMED and ADDS
are computed against the block rather than against each other, and
that the no-questions path still prints the block and its cost.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from forge import rag

_SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "rag_hot_tier.py"
_spec = importlib.util.spec_from_file_location("rag_hot_tier", _SCRIPT)
rag_hot_tier = importlib.util.module_from_spec(_spec)
sys.modules["rag_hot_tier"] = rag_hot_tier
sys.path.insert(0, str(_SCRIPT.parent))
_spec.loader.exec_module(rag_hot_tier)


def _run(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["rag_hot_tier", *argv])
    return rag_hot_tier.main()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "rag.db"
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(path))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    for kind, content in [
        ("fact", "Matériel : NiPoGi AM06PRO, Ryzen 5500U, 32 Go de RAM"),
        ("fact", "Possède un Steam Deck sous SteamOS"),
        ("history_summary", "user: Tu peux me lister mon matériel ? / assistant: Non."),
        ("decision", "Ne pas épingler les messages avec des emojis"),
    ]:
        rag.remember(conn, kind=kind, content=content, project=None)
    conn.close()
    return str(path)


def test_it_runs_and_prints_the_block_with_no_questions(db, capsys):

    sys.argv = ["rag_hot_tier", "--db", db]
    assert rag_hot_tier.main() == 0

    out = capsys.readouterr().out
    assert "NiPoGi" in out
    assert "HEADROOM" in out
    # A token count is not a cost anyone feels. The harness turns it
    # into the prefill seconds it actually buys -- and says that those
    # seconds are paid once, which three real runs established after
    # the branch had already shipped the opposite claim.
    assert "s of prefill on the first cold recall, then cached" in out


def test_an_empty_deliberate_store_is_the_finding(tmp_path, monkeypatch, capsys):
    path = tmp_path / "empty.db"
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(path))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    rag.remember(
        conn, kind="history_summary", content="user: x / assistant: y", project=None
    )
    conn.close()

    sys.argv = ["rag_hot_tier", "--db", str(path)]
    assert rag_hot_tier.main() == 1
    assert "nothing" in capsys.readouterr().out


def test_subsumed_and_adds_are_measured_against_the_block(db, capsys):
    sys.argv = [
        "rag_hot_tier",
        "--db",
        db,
        "--hit",
        "Tu peux me lister mon matériel ?",
        "--expect",
        "1",
    ]
    assert rag_hot_tier.main() == 0

    out = capsys.readouterr().out
    assert "SUBSUMED" in out and "ADDS" in out
    # #1 is a fact, so the block holds it whatever the search does --
    # which is the point of the tier and the reason this harness does
    # not report REACHED.
    assert "in the block" in out


def test_placeholders_are_refused(db, capsys):
    sys.argv = ["rag_hot_tier", "--db", db, "--hit", "<la question du 22/08>"]

    assert rag_hot_tier.main() == 1
    assert "placeholders" in capsys.readouterr().out


def test_the_rescue_arm_stays_asleep_without_a_cutoff(db, capsys, monkeypatch):
    """
    The expansion pass costs a model call per refused question, which
    is the one cost in this harness that is paid per question rather
    than once. It has to be asked for.
    """
    from forge.graphs import recall

    monkeypatch.setattr(
        recall,
        "_rescue",
        lambda q: pytest.fail("the rescue ran without --cutoff"),
    )
    _run(monkeypatch, ["--db", db, "--hit", "Quel processeur ?", "--expect", "-"])

    assert "expansion rescue" not in capsys.readouterr().out


def test_a_rescue_that_returns_nothing_is_not_reported_as_subsumed(
    db, capsys, monkeypatch
):
    """
    The distinction the first run of this arm got wrong, on the real
    store: the rescue applies the same cutoff to its own results, so
    on a store where everything sits beyond it, it brings back NOTHING
    -- which is a finding about the cutoff and not about the block
    absorbing anything.
    """
    from forge.graphs import recall

    monkeypatch.setattr(recall, "_drop_distant", lambda results, query: [])
    monkeypatch.setattr(recall, "_rescue", lambda q: [])

    _run(
        monkeypatch,
        ["--db", db, "--cutoff", "0.88", "--miss", "Comment s'appelle mon chat ?"],
    )

    out = capsys.readouterr().out
    assert "brought back NOTHING" in out
    assert "subsumes" not in out.split("--- expansion rescue")[1]


def test_what_the_rescue_adds_is_measured_against_the_block(db, capsys, monkeypatch):
    from forge.graphs import recall

    monkeypatch.setattr(recall, "_drop_distant", lambda results, query: [])
    monkeypatch.setattr(
        recall, "_rescue", lambda q: [{"id": 1, "distance": 0.5}, {"id": 3}]
    )

    _run(
        monkeypatch,
        ["--db", db, "--cutoff", "0.88", "--miss", "Comment s'appelle mon chat ?"],
    )

    out = capsys.readouterr().out
    # #1 is a deliberate entry and #3 is the archived transcript.
    assert "SUBSUMED      [1]" in out
    assert "ADDS          [3]" in out
