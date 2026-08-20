"""Tests for forge.graphs.research (search -> fetch top N -> synthesize).

This graph exists specifically to avoid depending on the router to
chain web_search into a synthesis/fetch step -- that reliably failed
live with the local model (see graphs/research.py's module docstring).
The whole point is that ONE call runs the full sequence
deterministically, so tests focus on: each node's behavior in
isolation, the conditional edges (no results -> error), and that a
partial fetch failure doesn't block synthesis.
"""

import forge.graphs.research as research_mod
from forge.graphs.research import build as build_research


def test_research_prompt_includes_todays_date(monkeypatch):
    from forge.context_info import today_line

    monkeypatch.setattr(
        research_mod.web_search,
        "search",
        lambda q: [{"title": "T", "url": "https://x.com", "content": "s"}],
    )
    monkeypatch.setattr(research_mod.web_fetch, "run", lambda url: "content")

    captured = {}

    def fake_call_llm(prompt, grammar=None):
        captured["prompt"] = prompt
        return "answer"

    monkeypatch.setattr(research_mod, "call_llm", fake_call_llm)
    build_research().run("query", initial_context={"query": "query"})

    assert today_line() in captured["prompt"]


def test_research_happy_path(monkeypatch):
    fake_results = [
        {
            "title": "Zig homepage",
            "url": "https://ziglang.org",
            "content": "A language.",
        },
        {
            "title": "Zig news",
            "url": "https://example.com/zig-news",
            "content": "Recent update.",
        },
    ]
    monkeypatch.setattr(research_mod.web_search, "search", lambda q: fake_results)
    monkeypatch.setattr(
        research_mod.web_fetch, "run", lambda url: f"Full content of {url}"
    )
    monkeypatch.setattr(
        research_mod,
        "call_llm",
        lambda p, grammar=None: "Zig est un langage moderne et rapide.",
    )

    state = build_research().run("Zig", initial_context={"query": "Zig"})

    assert state.ok
    assert state.final_output == "Zig est un langage moderne et rapide."
    assert state.final_tool == "research"


def test_research_no_results_goes_to_error_node(monkeypatch):
    monkeypatch.setattr(research_mod.web_search, "search", lambda q: [])

    state = build_research().run(
        "obscure query", initial_context={"query": "obscure query"}
    )

    assert state.ok  # error node surfaces as a message, not a crash
    assert "[no results]" in state.final_output


def test_research_search_failure_goes_to_error_node(monkeypatch):
    def raise_error(q):
        raise research_mod.web_search.SearchError("SearXNG down")

    monkeypatch.setattr(research_mod.web_search, "search", raise_error)

    state = build_research().run("query", initial_context={"query": "query"})

    assert state.ok
    assert "[error]" in state.final_output
    assert "SearXNG down" in state.final_output


def test_research_partial_fetch_failure_does_not_block_synthesis(monkeypatch):
    """A failed fetch on one of the top-N URLs must be skipped, not
    fatal -- synthesis should still proceed using whatever fetched
    successfully plus the search snippets."""
    fake_results = [
        {"title": "Good", "url": "https://good.com", "content": "snippet"},
        {"title": "Bad", "url": "https://bad.com", "content": "snippet"},
    ]
    monkeypatch.setattr(research_mod.web_search, "search", lambda q: fake_results)

    def fake_fetch(url):
        if url == "https://bad.com":
            return "[error] HTTP 404"
        return "Good page content"

    monkeypatch.setattr(research_mod.web_fetch, "run", fake_fetch)

    captured_prompt = {}

    def fake_call_llm(prompt, grammar=None):
        captured_prompt["prompt"] = prompt
        return "Synthesized answer."

    monkeypatch.setattr(research_mod, "call_llm", fake_call_llm)

    state = build_research().run("query", initial_context={"query": "query"})

    assert state.ok
    assert state.final_output == "Synthesized answer."
    assert "Good page content" in captured_prompt["prompt"]
    assert "bad.com" not in captured_prompt["prompt"]


def test_research_respects_fetch_top_n(monkeypatch):
    monkeypatch.setattr(research_mod, "RESEARCH_FETCH_TOP_N", 1)
    fake_results = [
        {"title": "One", "url": "https://one.com", "content": "s"},
        {"title": "Two", "url": "https://two.com", "content": "s"},
        {"title": "Three", "url": "https://three.com", "content": "s"},
    ]
    monkeypatch.setattr(research_mod.web_search, "search", lambda q: fake_results)

    fetched_urls = []

    def fake_fetch(url):
        fetched_urls.append(url)
        return f"content of {url}"

    monkeypatch.setattr(research_mod.web_fetch, "run", fake_fetch)
    monkeypatch.setattr(research_mod, "call_llm", lambda p, grammar=None: "answer")

    build_research().run("query", initial_context={"query": "query"})

    assert fetched_urls == ["https://one.com"]


def test_research_llm_unavailable(monkeypatch):
    from forge.errors import ProviderError

    monkeypatch.setattr(
        research_mod.web_search,
        "search",
        lambda q: [{"title": "T", "url": "https://x.com", "content": "s"}],
    )
    monkeypatch.setattr(research_mod.web_fetch, "run", lambda url: "content")
    monkeypatch.setattr(
        research_mod,
        "call_llm",
        lambda p, grammar=None: (_ for _ in ()).throw(ProviderError("down")),
    )

    state = build_research().run("query", initial_context={"query": "query"})

    assert not state.ok
    assert "LLM unavailable" in state.final_output


def test_research_unwraps_substantive_json_wrapped_answer(monkeypatch):
    """
    Regression test for the exact bug hit on research's first real
    run in production: the synthesis model wrapped a genuine,
    multi-sentence answer in {"tool":"chat","content":"..."} despite
    the prompt's explicit instruction and example not to. Unlike a
    degenerate echo (see test_research_cleans_json_wrapped_response_like_review),
    substantive content must be unwrapped to clean prose, not shown as
    raw JSON.
    """
    monkeypatch.setattr(
        research_mod.web_search,
        "search",
        lambda q: [{"title": "T", "url": "https://x.com", "content": "s"}],
    )
    monkeypatch.setattr(research_mod.web_fetch, "run", lambda url: "content")

    substantive_wrapped = (
        '{"tool":"chat","content":"Plusieurs sorties majeures sont '
        "attendues cette année, dont plusieurs titres AAA et des "
        'suites tres attendues par la communaute des joueurs."}'
    )
    monkeypatch.setattr(
        research_mod, "call_llm", lambda p, grammar=None: substantive_wrapped
    )

    state = build_research().run("query", initial_context={"query": "query"})

    assert state.final_output.startswith("Plusieurs sorties majeures")
    assert '"tool"' not in state.final_output


def test_research_cleans_json_wrapped_response_like_review(monkeypatch):
    """The synthesis prompt shares the same anti-JSON-habit risk as
    review's prompt (same underlying model) -- a degenerate JSON echo
    must be shown as-is, not silently unwrapped to something
    misleadingly short."""
    monkeypatch.setattr(
        research_mod.web_search,
        "search",
        lambda q: [{"title": "T", "url": "https://x.com", "content": "s"}],
    )
    monkeypatch.setattr(research_mod.web_fetch, "run", lambda url: "content")
    monkeypatch.setattr(
        research_mod,
        "call_llm",
        lambda p, grammar=None: '{"tool":"chat","content":"query"}',
    )

    state = build_research().run("query", initial_context={"query": "query"})

    assert state.final_output == '{"tool":"chat","content":"query"}'


def test_research_strips_think_blocks(monkeypatch):
    monkeypatch.setattr(
        research_mod.web_search,
        "search",
        lambda q: [{"title": "T", "url": "https://x.com", "content": "s"}],
    )
    monkeypatch.setattr(research_mod.web_fetch, "run", lambda url: "content")
    monkeypatch.setattr(
        research_mod,
        "call_llm",
        lambda p, grammar=None: "<think>thinking...</think>Final answer.",
    )

    state = build_research().run("query", initial_context={"query": "query"})

    assert state.final_output == "Final answer."
