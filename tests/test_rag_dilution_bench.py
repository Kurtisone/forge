"""
The dilution harness has to RUN before it can measure anything.

This exists because of a pattern already recorded four times on this
repository: a measurement harness that fails, or measures something
slightly beside the point, and costs a real round trip on the Deck to
find out. bench/rag_dilution.py is meant to be copied into the
container and run once against the live embedding server; a crash two
hundred lines in wastes that trip.

So the plumbing is exercised here with a stub embedder -- the numbers
it produces are meaningless, and asserting on them would be the same
mistake in a different direction. What is asserted is that the three
shapes get planted, the needle is found in each, and the verdict path
is reached.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from forge import rag

_SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "rag_dilution.py"
_spec = importlib.util.spec_from_file_location("rag_dilution", _SCRIPT)
rag_dilution = importlib.util.module_from_spec(_spec)
sys.modules["rag_dilution"] = rag_dilution
_spec.loader.exec_module(rag_dilution)


def _bag_of_words_embedding(text: str) -> list[float]:
    """
    A stand-in that at least varies with the text, so ranking is not
    decided by insertion order. Not a model: a hashed word count,
    L2-normalised the way llama-server normalises what it returns.
    """
    vector = [0.0] * rag.EMBEDDING_DIM
    for word in text.lower().split():
        vector[hash(word) % rag.EMBEDDING_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "dilution.db"))
    monkeypatch.setattr(rag, "_embed", _bag_of_words_embedding)
    return tmp_path / "dilution.db"


def _run(db, *flags):
    argv = ["rag_dilution.py", "--db", str(db), *flags]
    saved, sys.argv = sys.argv, argv
    try:
        return rag_dilution.main()
    finally:
        sys.argv = saved


def test_the_harness_runs_and_reaches_a_verdict(stubbed, capsys):
    assert _run(stubbed) == 0

    out = capsys.readouterr().out
    assert "coût de l'enfouissement" in out
    assert "plancher" in out


def test_all_three_shapes_are_planted(stubbed):
    _run(stubbed)

    conn = rag.get_connection()
    try:
        projects = {e["project"] for e in rag.list_entries(conn, limit=100)}
    finally:
        conn.close()

    assert projects == {"alone", "buried", "split"}


def test_the_split_shape_holds_one_entry_per_exchange(stubbed):
    _run(stubbed)

    conn = rag.get_connection()
    try:
        split = rag.list_entries(conn, project="split", limit=100)
        buried = rag.list_entries(conn, project="buried", limit=100)
    finally:
        conn.close()

    assert len(split) == len(rag_dilution.FILLER) + 1
    assert len(buried) == 1


def test_the_needle_is_findable_in_every_shape(stubbed):
    """
    The comparison is only meaningful if the sentence is present three
    times. _rank_of_needle returning 0 anywhere means the harness is
    measuring something else.
    """
    _run(stubbed)

    conn = rag.get_connection()
    try:
        for shape in ("alone", "buried", "split"):
            entries = rag.list_entries(conn, project=shape, limit=100)
            assert any(rag_dilution.NEEDLE in e["content"] for e in entries), shape
    finally:
        conn.close()


def test_it_refuses_to_plant_in_the_real_store(capsys):
    assert _run("data/forge_rag.db") == 1
    assert "refusing" in capsys.readouterr().out


def test_the_question_is_not_a_copy_of_the_sentence():
    """
    Matching a paraphrase is the job; matching a copy measures the
    string, not the embedding. The first --no-plant run of
    bench/recall_distance.py went wrong in exactly this way.
    """
    question_words = set(rag_dilution.QUESTION.lower().rstrip("?").split())
    needle_words = set(rag_dilution.NEEDLE.lower().split())

    assert not question_words <= needle_words
