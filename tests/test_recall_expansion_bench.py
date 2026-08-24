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
        expansion, "call_llm", lambda prompt, grammar=None: '["Quel processeur ?"]'
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


def test_repeating_keeps_the_worst_draw_for_a_hit(store, monkeypatch):
    """
    Worst means FARTHEST for a named entry -- the draw where the
    rescue is least likely to reach it.
    """
    from forge import expansion, rag

    monkeypatch.setattr(expansion, "variants", lambda q, mode: ["une reformulation"])
    draws = iter([[{"id": 1, "distance": 0.70}], [{"id": 1, "distance": 0.95}]])
    monkeypatch.setattr(
        rag, "search_many", lambda conn, queries, top_k, exclude_kind=None: next(draws)
    )

    best = recall_expansion._collect(
        None, [(("hit", 0), "une question", "1")], ["terms"], 5, 2, 0.88
    )

    assert best[(("hit", 0), "terms")][1][0]["distance"] == 0.95


def test_repeating_keeps_the_worst_draw_for_a_miss(store, monkeypatch):
    """The same rule from the other end: nearest, the draw most likely
    to break a refusal."""
    from forge import expansion, rag

    monkeypatch.setattr(expansion, "variants", lambda q, mode: ["une reformulation"])
    draws = iter([[{"id": 9, "distance": 1.20}], [{"id": 9, "distance": 0.91}]])
    monkeypatch.setattr(
        rag, "search_many", lambda conn, queries, top_k, exclude_kind=None: next(draws)
    )

    best = recall_expansion._collect(
        None, [(("miss", 0), "une question", None)], ["terms"], 5, 2, 0.88
    )

    assert best[(("miss", 0), "terms")][1][0]["distance"] == 0.91


def test_a_draw_with_no_usable_rewrites_counts_as_the_worst(store, monkeypatch):
    """
    It is what the deployment would have done that time: no rescue at
    all. Silently preferring the draws that produced rewrites would
    measure a mechanism nobody runs.
    """
    from forge import expansion, rag

    calls = iter([[], ["une reformulation"]])
    monkeypatch.setattr(expansion, "variants", lambda q, mode: next(calls))
    monkeypatch.setattr(
        rag,
        "search_many",
        lambda conn, queries, top_k, exclude_kind=None: [{"id": 1, "distance": 0.1}],
    )

    best = recall_expansion._collect(
        None, [(("hit", 0), "une question", "1")], ["llm"], 5, 2, 0.88
    )

    assert best[(("hit", 0), "llm")][0] == []
    assert best[(("hit", 0), "llm")][1] == []


def test_the_draws_are_interleaved_not_repeated_back_to_back(store, monkeypatch):
    """
    Three consecutive calls on llama.cpp measure nothing: the first
    warms the prompt cache and the next two are cache hits returning
    byte-identical output -- prompt_ms 4116, then 189, then 186, same
    107 characters back each time (2026-08-24).

    The variance lives BETWEEN cache states, so a draw has to follow a
    different predecessor to be a different draw.
    """
    from forge import expansion, rag

    asked = []
    monkeypatch.setattr(
        expansion,
        "variants",
        lambda q, mode: (asked.append(q), ["une reformulation"])[1],
    )
    monkeypatch.setattr(
        rag, "search_many", lambda conn, queries, top_k, exclude_kind=None: []
    )

    recall_expansion._collect(
        None,
        [
            (("hit", 0), "première question", "1"),
            (("miss", 0), "deuxième question", None),
        ],
        ["terms"],
        5,
        2,
        0.88,
    )

    assert asked == [
        "première question",
        "deuxième question",
        "première question",
        "deuxième question",
    ]


def test_a_rescue_that_answers_with_another_entry_is_counted(store, monkeypatch):
    """
    The outcome the counts had no name for. On 2026-08-24 the rescue
    put #307 at 0.9435 while something else came in at 0.8777 --
    inside the cutoff, ahead of it. In the deployment the question
    stops being refused and starts being answered out of the wrong
    entry, and the verdict printed FALSE RESCUES 0 because it only
    looked at the misses.
    """
    rows = [
        {"id": 212, "distance": 0.8777, "content": "un échange archivé"},
        {"id": 307, "distance": 0.9435, "content": "Matériel : NiPoGi"},
    ]

    found = recall_expansion._intruder(rows, "307", 0.88)

    assert found["id"] == 212


def test_the_named_entry_is_never_its_own_intruder(store):
    rows = [{"id": 307, "distance": 0.72, "content": "Matériel : NiPoGi"}]

    assert recall_expansion._intruder(rows, "307", 0.88) is None


def test_nothing_within_the_cutoff_is_no_intruder(store):
    rows = [{"id": 212, "distance": 0.95, "content": "loin"}]

    assert recall_expansion._intruder(rows, "307", 0.88) is None


def test_the_worst_draw_is_the_one_that_answers_wrong(store, monkeypatch):
    """
    Measured 2026-08-24, three interleaved draws of the hardware
    question:

        draw A   #307 at 0.9435, nothing else within the cutoff
        draw C   #307 at 0.9766, #167 at 0.8777 within the cutoff

    Ranked by distance to the named entry, draw A is farther and
    "wins" as the worst -- and the verdict printed WRONG ENTRY 0. Draw
    C is the one where the deployment answers out of an archived
    Containerfile dump. A single-draw run found it by accident and the
    three-draw run buried it: a sampling rule that makes more
    measurement less informative is the wrong rule.
    """
    from forge import expansion, rag

    monkeypatch.setattr(expansion, "variants", lambda q, mode: ["une reformulation"])
    draws = iter(
        [
            [{"id": 307, "distance": 0.9435}],
            [{"id": 307, "distance": 0.9766}, {"id": 167, "distance": 0.8777}],
        ]
    )
    monkeypatch.setattr(
        rag, "search_many", lambda conn, queries, top_k, exclude_kind=None: next(draws)
    )

    best = recall_expansion._collect(
        None, [(("hit", 0), "une question", "307")], ["terms"], 5, 2, 0.88
    )

    assert any(r["id"] == 167 for r in best[(("hit", 0), "terms")][1])


def test_a_draw_that_finds_the_entry_beats_one_that_refuses(store):
    """Worst by outcome: a refusal is bad, a wrong answer is worse,
    and finding the entry is what we were after."""
    found = [{"id": 307, "distance": 0.72}]
    refusal = [{"id": 307, "distance": 1.4}]
    wrong = [{"id": 167, "distance": 0.80}]

    assert recall_expansion._badness(wrong, "307", 0.88) > recall_expansion._badness(
        refusal, "307", 0.88
    )
    assert recall_expansion._badness(refusal, "307", 0.88) > recall_expansion._badness(
        found, "307", 0.88
    )


def test_the_nearest_other_row_is_found_at_any_distance(store):
    """
    Not the same question as _intruder, which asks what gets in under
    the first pass's cutoff. At any threshold admitting the answer,
    every row nearer than it is admitted first -- so a regime block
    scoring only the named entries measures a search with no
    competitors in it.

    Measured 2026-08-24: #314 came back at 1.0480 with something at
    0.9618 ahead of it. Nothing was within 0.88, so WRONG ENTRY was 0
    and the regime printed a gap of +0.1032 that no threshold could
    actually deliver.
    """
    rows = [
        {"id": 314, "distance": 1.0480, "content": "services podman"},
        {"id": 313, "distance": 0.9618, "content": "Steam Deck, SteamOS"},
    ]

    assert recall_expansion._nearest_other(rows, "314")["id"] == 313


def test_the_named_entry_is_never_its_own_rival(store):
    rows = [{"id": 307, "distance": 0.72, "content": "Matériel : NiPoGi"}]

    assert recall_expansion._nearest_other(rows, "307") is None
