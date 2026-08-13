"""Tests for forge.graphs.recall (memory search -> synthesize).

This graph exists specifically to avoid depending on the router to
chain memory:recall into a synthesis step -- that reliably failed
live with the local model (see graphs/recall.py's module docstring),
the same failure already fixed for web_search by graphs/research.py.
Tests focus on: each node's behavior in isolation, the conditional
edge (no results -> error), and the synthesis prompt's output
cleaning (shared reasoning with research.py's tests, same underlying
model and the same class of prompt-leak risk).
"""

import forge.graphs.recall as recall_mod
from forge import rag
from forge.graphs.recall import build as build_recall


def test_recall_happy_path(monkeypatch):
    fake_results = [
        {"kind": "fact", "content": "Possède un Steam Deck", "project": None},
        {"kind": "fact", "content": "Possède un Dell R710", "project": None},
    ]
    monkeypatch.setattr(recall_mod.memory_tool, "search", lambda q: fake_results)
    monkeypatch.setattr(
        recall_mod, "call_llm", lambda p: "Tu as un Steam Deck et un Dell R710."
    )

    state = build_recall().run(
        "Tu peux me lister mon matériel ?",
        initial_context={"query": "Tu peux me lister mon matériel ?"},
    )

    assert state.ok
    assert state.final_output == "Tu as un Steam Deck et un Dell R710."
    assert state.final_tool == "recall"


def test_recall_no_results_goes_to_error_node(monkeypatch):
    monkeypatch.setattr(recall_mod.memory_tool, "search", lambda q: [])

    state = build_recall().run(
        "obscure query", initial_context={"query": "obscure query"}
    )

    assert state.ok  # error node surfaces as a message, not a crash
    assert "[no memory]" in state.final_output


def test_recall_embedding_failure_goes_to_error_node(monkeypatch):
    def raise_error(q):
        raise rag.EmbeddingError("400 Bad Request")

    monkeypatch.setattr(recall_mod.memory_tool, "search", raise_error)

    state = build_recall().run("query", initial_context={"query": "query"})

    assert state.ok
    assert "[error]" in state.final_output
    assert "400 Bad Request" in state.final_output


def test_recall_prompt_includes_ranked_clipped_entries(monkeypatch):
    """
    The synthesis prompt must be built from the same ranked/clipped
    formatting tools/memory.py already uses for direct dispatch
    (format_results), not a re-derived one -- so the two callers can't
    silently drift into two different notions of a memory hit (see
    format_results' docstring in tools/memory.py).
    """
    fake_results = [
        {"kind": "history_summary", "content": "bavardage " * 200, "project": None},
        {"kind": "fact", "content": "Possède un Steam Deck", "project": None},
    ]
    monkeypatch.setattr(recall_mod.memory_tool, "search", lambda q: fake_results)
    monkeypatch.setattr(recall_mod.memory_tool, "MEMORY_RECALL_MAX_CHARS", 50)

    captured = {}

    def fake_call_llm(prompt):
        captured["prompt"] = prompt
        return "answer"

    monkeypatch.setattr(recall_mod, "call_llm", fake_call_llm)
    build_recall().run("query", initial_context={"query": "query"})

    # fact ranked before history_summary in the prompt block
    assert captured["prompt"].index("[fact]") < captured["prompt"].index(
        "[history_summary]"
    )


def test_recall_llm_unavailable(monkeypatch):
    from forge.errors import ProviderError

    monkeypatch.setattr(
        recall_mod.memory_tool,
        "search",
        lambda q: [{"kind": "fact", "content": "x", "project": None}],
    )
    monkeypatch.setattr(
        recall_mod, "call_llm", lambda p: (_ for _ in ()).throw(ProviderError("down"))
    )

    state = build_recall().run("query", initial_context={"query": "query"})

    assert not state.ok
    assert "LLM unavailable" in state.final_output


def test_recall_unwraps_substantive_json_wrapped_answer(monkeypatch):
    """
    Same failure class as research's first real run (see
    test_research.py::test_research_unwraps_substantive_json_wrapped_answer):
    a genuine answer wrapped in {"tool":"chat","content":"..."} despite
    the prompt's explicit instruction not to. Substantive content must
    be unwrapped to clean prose.
    """
    monkeypatch.setattr(
        recall_mod.memory_tool,
        "search",
        lambda q: [
            {"kind": "fact", "content": "Possède un Steam Deck", "project": None}
        ],
    )
    substantive_wrapped = (
        '{"tool":"chat","content":"Tu as un Steam Deck, d\\u2019après ce '
        'que tu m\\u2019as dit précédemment sur ton matériel."}'
    )
    monkeypatch.setattr(recall_mod, "call_llm", lambda p: substantive_wrapped)

    state = build_recall().run("query", initial_context={"query": "query"})

    assert state.final_output.startswith("Tu as un Steam Deck")
    assert '"tool"' not in state.final_output


def test_recall_cleans_degenerate_json_echo(monkeypatch):
    """A degenerate JSON echo must be shown as-is, not silently
    unwrapped to something misleadingly short -- same reasoning as
    research's equivalent test."""
    monkeypatch.setattr(
        recall_mod.memory_tool,
        "search",
        lambda q: [{"kind": "fact", "content": "x", "project": None}],
    )
    monkeypatch.setattr(
        recall_mod, "call_llm", lambda p: '{"tool":"chat","content":"query"}'
    )

    state = build_recall().run("query", initial_context={"query": "query"})

    assert state.final_output == '{"tool":"chat","content":"query"}'


def test_recall_strips_think_blocks(monkeypatch):
    monkeypatch.setattr(
        recall_mod.memory_tool,
        "search",
        lambda q: [{"kind": "fact", "content": "x", "project": None}],
    )
    monkeypatch.setattr(
        recall_mod, "call_llm", lambda p: "<think>thinking...</think>Final answer."
    )

    state = build_recall().run("query", initial_context={"query": "query"})

    assert state.final_output == "Final answer."


def test_recall_answer_is_capped():
    from forge.config import RECALL_MAX_ANSWER_CHARS

    cleaned = recall_mod._clean_synthesis_response(
        "x" * (RECALL_MAX_ANSWER_CHARS + 500)
    )

    assert len(cleaned) <= RECALL_MAX_ANSWER_CHARS + 1
    assert cleaned.endswith("…")
