"""
The expansion harness has to RUN before it can measure anything.

Same reasoning as tests/test_rag_dilution_bench.py, and the same
pattern behind it: a harness that crashes two hundred lines in costs a
real round trip on the Deck to find out. The embedder is a stub and
the model is a stub, so the numbers here are meaningless and asserting
on them would be the same mistake pointed the other way. What is
asserted is that every path is reachable -- the four outcomes a hit
can have, the false rescue, and the refusals that keep a bad run from
producing a confident verdict.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

from forge import expansion, rag

_SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "recall_expansion.py"
sys.path.insert(0, str(_SCRIPT.parent))
_spec = importlib.util.spec_from_file_location("recall_expansion", _SCRIPT)
recall_expansion = importlib.util.module_from_spec(_spec)
sys.modules["recall_expansion"] = recall_expansion
_spec.loader.exec_module(recall_expansion)


def _bag_of_words_embedding(text: str) -> list[float]:
    """
    A stand-in that at least varies with the text. Not a model: a
    hashed word count, L2-normalised the way llama-server normalises
    what it returns.
    """
    vector = [0.0] * rag.EMBEDDING_DIM
    for word in text.lower().replace("?", " ").split():
        vector[hash(word) % rag.EMBEDDING_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "_embed", _bag_of_words_embedding)
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "bench.db"))
    monkeypatch.setattr(
        expansion, "call_llm", lambda prompt, grammar=None: '["processeur mémoire"]'
    )

    conn = rag.get_connection()
    try:
        rag.remember(
            conn,
            kind="fact",
            content="Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM",
            project=None,
        )
    finally:
        conn.close()
    return tmp_path / "bench.db"


def _run(db, *flags):
    argv = ["recall_expansion.py", "--db", str(db), *flags]
    saved, sys.argv = sys.argv, argv
    try:
        return recall_expansion.main()
    finally:
        sys.argv = saved


def test_it_runs_and_reaches_a_verdict(store, capsys):
    assert (
        _run(
            store,
            "--cutoff",
            "0.88",
            "--hit",
            "Tu peux me lister mon matériel ?",
            "--expect",
            "1",
            "--miss",
            "Comment s'appelle mon chat ?",
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "VERDICT" in out
    assert "FALSE RESCUES" in out


def test_the_variants_are_printed_next_to_the_numbers(store, capsys):
    """
    A column of distances nobody can attribute is unreadable: the
    question about a rescue is always WHICH rephrasing produced it.
    """
    _run(store, "--cutoff", "0.88", "--hit", "Tu peux me lister mon matériel ?")

    assert "lister mon matériel" in capsys.readouterr().out


def test_one_mode_can_be_asked_for_alone(store, capsys):
    """--mode terms is free; --mode llm costs a model call per question."""
    _run(
        store,
        "--cutoff",
        "0.88",
        "--mode",
        "terms",
        "--hit",
        "Tu peux me lister mon matériel ?",
    )

    out = capsys.readouterr().out
    assert "TERMS" in out
    assert "LLM" not in out


def test_without_expect_the_hit_verdict_is_suppressed(store, capsys):
    """
    This store holds archived exchanges that contain the question and
    outrank the answer to it. Scoring hits on whatever came back first
    is the fault that produced the finest number recall_distance ever
    printed.
    """
    _run(store, "--cutoff", "0.88", "--hit", "Tu peux me lister mon matériel ?")

    assert "No --expect given" in capsys.readouterr().out


def test_placeholders_are_refused(store, capsys):
    assert _run(store, "--cutoff", "0.88", "--hit", "<la question du 22/08>") == 1
    assert "placeholders" in capsys.readouterr().out


def test_a_mismatched_expect_count_is_refused(store, capsys):
    assert (
        _run(
            store,
            "--cutoff",
            "0.88",
            "--hit",
            "une question",
            "--expect",
            "1",
            "--expect",
            "2",
        )
        == 1
    )
    assert "--expect given" in capsys.readouterr().out


def test_no_cutoff_anywhere_is_refused(store, capsys, monkeypatch):
    """
    The rescue pass is DEFINED by the cutoff -- it runs when the cutoff
    drops everything. Scoring it against a default nobody measured
    would be a verdict about a configuration that does not exist.
    """
    monkeypatch.setattr("forge.config.RECALL_MAX_DISTANCE", None)

    assert _run(store, "--hit", "une question", "--expect", "1") == 1
    assert "no cutoff" in capsys.readouterr().out


def test_a_missing_store_is_refused(tmp_path, capsys):
    assert _run(tmp_path / "nowhere.db", "--cutoff", "0.88", "--hit", "q") == 1
    assert "does not exist" in capsys.readouterr().out


def test_the_rescue_regime_is_reported_separately(store, capsys):
    """
    A variant is a short phrase and the question it replaced was a
    sentence, so the whole distribution moves. Scoring the rescue pass
    with the first-pass cutoff compares it against a scale it was not
    measured on -- the fault docs/memory.md already names about the
    threshold itself.
    """
    _run(
        store,
        "--cutoff",
        "0.88",
        "--mode",
        "llm",
        "--hit",
        "Tu peux me lister mon matériel ?",
        "--expect",
        "1",
        "--miss",
        "Comment s'appelle mon chat ?",
    )

    out = capsys.readouterr().out
    assert "the rescue regime" in out
    assert "gap" in out


def test_it_refuses_to_name_a_cutoff_on_two_questions(store, capsys):
    """
    Three a side, as recall_distance asks for, and for the same
    reason: a gap over two numbers is an anecdote with a decimal point
    on it.
    """
    _run(
        store,
        "--cutoff",
        "0.88",
        "--mode",
        "llm",
        "--hit",
        "Tu peux me lister mon matériel ?",
        "--expect",
        "1",
        "--miss",
        "Comment s'appelle mon chat ?",
    )

    out = capsys.readouterr().out
    assert "anecdote" in out or "overlap" in out


def test_a_named_entry_that_never_came_back_is_not_a_distance(store):
    """
    read_row falls back to the closest row so the table always has a
    number in it. That fallback is exactly wrong for a verdict: an
    entry that never came back is not an entry at a distance.
    """
    rows = [{"id": 7, "distance": 0.4}]

    assert recall_expansion._distance_of(rows, "308") is None
    assert recall_expansion._distance_of(rows, "7") == 0.4


def test_a_question_with_no_named_entry_gets_its_candidates_printed(store, capsys):
    """
    The harness demanded an --expect id and offered no way to find
    one, which stopped a real measurement on 2026-08-24. The first run
    is what finds the ids.
    """
    _run(
        store,
        "--cutoff",
        "0.88",
        "--mode",
        "terms",
        "--hit",
        "Tu peux me lister mon matériel ?",
    )

    out = capsys.readouterr().out
    assert "candidates" in out
    assert "NiPoGi" in out


def test_a_dash_means_not_known_yet(store, capsys):
    """
    Different from omitting --expect entirely: it lets the ids you DO
    have stay aligned with their questions.
    """
    assert (
        _run(
            store,
            "--cutoff",
            "0.88",
            "--mode",
            "terms",
            "--hit",
            "Tu peux me lister mon matériel ?",
            "--expect",
            "1",
            "--hit",
            "Une question dont j'ignore la réponse",
            "--expect",
            "-",
        )
        == 0
    )

    assert "candidates" in capsys.readouterr().out


def test_the_misalignment_message_says_how_to_fix_it(store, capsys):
    _run(
        store,
        "--cutoff",
        "0.88",
        "--hit",
        "une question",
        "--hit",
        "une autre",
        "--expect",
        "1",
    )

    assert "Pass - for the ones" in capsys.readouterr().out
