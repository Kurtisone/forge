"""
Smoke tests for the orchestrator. No network: the LLM call is
monkeypatched at the boundary (forge.orchestrator.call_llm).
"""

import json

import forge.orchestrator as orch_mod
from forge.orchestrator import Orchestrator


def test_chat_round_trip(monkeypatch):
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "chat", "content": "hi there"}),
    )
    result = Orchestrator().run("hello")
    assert result.ok
    assert result.tool == "chat"
    assert result.output == "hi there"
    assert result.steps == 1


def test_dispatch_attaches_sub_steps_published_by_graph_based_tools(monkeypatch):
    """Graph-based tools (review/research/sysadmin) can publish their
    internal node steps via forge.subtrace right before returning --
    see that module's docstring. The orchestrator must pick them up
    and attach them to the step's TraceStep without changing the
    str-only tool contract for every other tool."""
    from forge import subtrace

    def fake_tool(content):
        subtrace.publish(
            [{"label": "discover", "detail": "1 unit", "ok": True, "duration_ms": 5}]
        )
        return "diagnosis text"

    monkeypatch.setattr(orch_mod, "get_tool", lambda name: fake_tool)
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "sysadmin", "content": "{}"}),
    )

    result = Orchestrator().run("le service plante")

    assert result.ok
    assert result.trace[0].sub_steps == [
        {"label": "discover", "detail": "1 unit", "ok": True, "duration_ms": 5}
    ]


def test_dispatch_sub_steps_do_not_leak_to_next_tool(monkeypatch):
    """subtrace.pop() must clear the channel -- a tool that doesn't
    publish anything must never inherit a previous tool's steps."""
    from forge import subtrace

    subtrace.publish(
        [{"label": "stale", "detail": "leftover", "ok": True, "duration_ms": 1}]
    )

    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "chat", "content": "hi"}),
    )
    result = Orchestrator().run("hello")

    assert result.trace[0].sub_steps is None


def test_end_to_end_router_reaches_shell_when_enabled(monkeypatch, tmp_path):
    """
    Full stack, not mocked at the parser boundary this time: with
    ENABLED_TOOLS actually including "shell", a router decision naming
    "shell" must survive parsing and reach the real shell tool --
    this is the v3.5 change (previously the router's own validation
    hardcoded {"chat", "code"} regardless of ENABLED_TOOLS, so "shell"
    would have been silently downgraded to "chat" before ever
    reaching dispatch).
    """
    import forge.config as cfg
    import forge.tools.registry as registry_mod
    import forge.tools.shell as shell_mod

    monkeypatch.setattr(cfg, "ENABLED_TOOLS", {"chat", "code", "shell"})
    monkeypatch.setattr(registry_mod, "ENABLED_TOOLS", {"chat", "code", "shell"})
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_ALLOWED_COMMANDS", {"echo"})
    monkeypatch.setattr(cfg, "SHELL_TIMEOUT", 10)
    monkeypatch.setattr(shell_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(shell_mod, "SHELL_ALLOWED_COMMANDS", {"echo"})
    monkeypatch.setattr(shell_mod, "SHELL_TIMEOUT", 10)
    registry_mod.load_tools()

    try:
        monkeypatch.setattr(
            orch_mod,
            "call_llm",
            lambda prompt: json.dumps(
                {"tool": "shell", "content": "echo hi-from-shell"}
            ),
        )
        result = Orchestrator().run("run echo hi-from-shell")

        assert result.ok
        assert result.tool == "shell"
        assert "hi-from-shell" in result.output
    finally:
        # Undo every monkeypatch made in this test (ENABLED_TOOLS,
        # WORKSPACE_DIR, etc.) BEFORE reloading, so load_tools() runs
        # against the real config -- not the mocked one still in
        # effect during a plain `finally`, which would otherwise leave
        # the shared, module-level TOOLS registry permanently pointed
        # at this test's temporary tool set for every test after it.
        monkeypatch.undo()
        registry_mod.load_tools()


def test_code_round_trip(monkeypatch):
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "code", "content": "print(1)"}),
    )
    result = Orchestrator().run("write code")
    assert result.ok
    assert result.tool == "code"
    assert "print(1)" in result.output


def test_malformed_router_output_falls_back_to_chat(monkeypatch):
    monkeypatch.setattr(orch_mod, "call_llm", lambda prompt: "not json at all")
    result = Orchestrator().run("hello")
    assert result.ok
    assert result.tool == "chat"
    assert result.output == "not json at all"


def test_runaway_local_model_output_still_parses(monkeypatch):
    """
    Regression test for a real-world failure: a local llama.cpp model
    kept generating fake "User: ..." turns after the JSON object
    because the old stop sequences never matched. The parser must
    extract the leading JSON object instead of discarding the whole
    answer just because of trailing garbage.
    """
    raw = (
        '{"tool":"chat","content":"answer"}\n'
        "User: some question\n"
        '{"tool":"chat","content":"answer"}\n'
        "User: some question\n"
    )
    monkeypatch.setattr(orch_mod, "call_llm", lambda prompt: raw)
    result = Orchestrator().run("some question")
    assert result.ok
    assert result.tool == "chat"
    assert result.output == "answer"


def test_leaked_role_prefix_is_stripped(monkeypatch):
    """
    Regression test: once conversation history was added to the
    router prompt, some local models started leaking 'Assistant: ...'
    style prefixes into otherwise-non-JSON output. The label must not
    end up visible in the final answer.
    """
    monkeypatch.setattr(
        orch_mod, "call_llm", lambda prompt: "Assistant: here is my answer"
    )
    result = Orchestrator().run("hello")
    assert result.ok
    assert result.output == "here is my answer"


def test_persisted_user_turn_is_rendered_exactly_like_the_live_turn(monkeypatch):
    """
    End-to-end half of the pure-append invariant, through the real
    persistence path rather than a hand-built history list: what
    orchestrator._finish() writes to memory.json must come back out of
    _format_history() rendered byte-for-byte like the live "User:" line
    that carried it in the turn before.

    This replaces test_history_is_passed_as_context_not_dialogue, which
    asserted the opposite ("\nUser: ...\n" must NOT appear) for the
    bullet-summary format. That format existed because a full
    'User: ... / Assistant: ...' dialogue made local models continue the
    conversation in prose instead of emitting JSON. Only the user half
    is symmetric now; assistant turns still render as a parenthesised
    aside, so no bare "User:" line in this prompt is ever followed by
    anything but JSON. (Memory file isolation comes from the autouse
    fixture in conftest.py.)
    """
    captured = {}

    def capture_and_answer(prompt):
        captured["prompt"] = prompt
        return json.dumps({"tool": "chat", "content": "ok"})

    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "chat", "content": "Salut Alexandre !"}),
    )
    Orchestrator().run("Je m'appelle Alexandre")

    monkeypatch.setattr(orch_mod, "call_llm", capture_and_answer)
    Orchestrator().run("Comment je m'appelle ?")

    from forge.router.prompt import render_user_turn

    assert render_user_turn("Je m'appelle Alexandre") in captured["prompt"]
    # The assistant side stays deliberately asymmetric.
    assert "\nAssistant: Salut Alexandre !" not in captured["prompt"]
    assert "(you answered: Salut Alexandre !)" in captured["prompt"]


def test_unknown_tool_falls_back_to_chat(monkeypatch):
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "shell", "content": "rm -rf /"}),
    )
    result = Orchestrator().run("do something dangerous")
    assert (
        result.tool == "chat"
    )  # shell isn't registered, parser/orchestrator fall back


def test_fallback_placeholder_is_not_remembered(monkeypatch, tmp_path):
    """
    The bug this locks down: a router failure (empty/garbled output)
    produces a placeholder chat response, dispatch succeeds trivially
    (chat's tool just echoes content), so result.ok is True -- and
    before this fix, MEMORY_ENABLED and result.ok was the only gate,
    so the placeholder got saved as a real assistant turn. The next
    prompt would then include it as context, which can make a model
    that got confused once more likely to get confused again on the
    very next turn -- an escalating failure loop, seen in the wild as
    repeated 'router output was empty' warnings.
    """
    import forge.config as cfg
    import forge.memory as memory_mod

    monkeypatch.setattr(cfg, "MEMORY_ENABLED", True)
    monkeypatch.setattr(orch_mod, "MEMORY_ENABLED", True)
    monkeypatch.setattr(memory_mod, "MEMORY_FILE", str(tmp_path / "memory.json"))

    # Empty raw output -> the parser's empty-output placeholder path.
    monkeypatch.setattr(orch_mod, "call_llm", lambda prompt: "   ")

    result = Orchestrator().run("hello")

    assert result.ok  # dispatch of "chat" always succeeds, even for a placeholder
    assert "Je n'ai pas pu générer" in result.output
    assert memory_mod.get_history() == []  # must NOT have been remembered


def test_real_chat_answer_is_still_remembered(monkeypatch, tmp_path):
    """Sanity check alongside the test above: a genuine answer must
    still be persisted -- the fix should only skip placeholders, not
    memory as a whole."""
    import forge.config as cfg
    import forge.memory as memory_mod

    monkeypatch.setattr(cfg, "MEMORY_ENABLED", True)
    monkeypatch.setattr(orch_mod, "MEMORY_ENABLED", True)
    monkeypatch.setattr(memory_mod, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "chat", "content": "hi there"}),
    )

    Orchestrator().run("hello")

    history = memory_mod.get_history()
    assert any(h["content"] == "hi there" for h in history)


def test_long_tool_output_is_persisted_in_full(monkeypatch, tmp_path):
    """
    Regression test: _remember() used to hard-truncate both sides of
    an exchange to 300 chars before persisting. That was originally
    meant to keep the router's own prompt from ballooning on large
    pastes, but the v3.9 web UI renders GET /history directly, so a
    long tool result (e.g. reading a real file via the `files` tool)
    showed up cut off mid-word on screen instead of just producing a
    shorter prompt on the next turn. Full content must round-trip.
    """
    import forge.config as cfg
    import forge.memory as memory_mod

    monkeypatch.setattr(cfg, "MEMORY_ENABLED", True)
    monkeypatch.setattr(orch_mod, "MEMORY_ENABLED", True)
    monkeypatch.setattr(memory_mod, "MEMORY_FILE", str(tmp_path / "memory.json"))

    long_answer = "line\n" * 200  # 1000 chars, well past the old 300-char cap
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "chat", "content": long_answer}),
    )

    Orchestrator().run("read this file")

    history = memory_mod.get_history()
    assert any(h["content"] == long_answer for h in history)


def test_read_then_write_flow_actually_updates_the_file(monkeypatch, tmp_path):
    """
    End-to-end version of the router-prompt tests: a 2-step run where
    step 1 reads an existing file (done:false) and step 2 writes the
    modified version back must actually update the file on disk, not
    just produce a plausible-looking chat answer. This is the real
    failure this whole feature targets -- observed live, several
    "remplace X par Y dans hello.go" requests in a row all answered
    with the original unmodified content, having never called the
    files tool at all.
    """
    import forge.config as cfg
    from forge.tools import files as files_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_tool, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setitem(TOOLS, "files", files_tool.run)

    (tmp_path / "hello.go").write_text(
        'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("Hello, World!")\n}\n'
    )

    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "tool": "files",
                    "content": json.dumps({"action": "read", "path": "hello.go"}),
                    "done": False,
                }
            )
        new_content = 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("Bienvenue")\n}\n'
        return json.dumps(
            {
                "tool": "files",
                "content": json.dumps(
                    {"action": "write", "path": "hello.go", "content": new_content}
                ),
            }
        )

    monkeypatch.setattr(orch_mod, "call_llm", fake_llm)

    result = Orchestrator(max_steps=2).run("remplace Hello World par Bienvenue")

    assert result.ok
    assert calls["n"] == 2
    on_disk = (tmp_path / "hello.go").read_text()
    assert "Bienvenue" in on_disk
    assert "Hello, World!" not in on_disk
    assert "```diff" in result.output  # a real diff, not a bare confirmation


def test_multi_step_run_persists_exactly_one_clean_exchange(monkeypatch, tmp_path):
    """
    Regression test for the v3.8 cache-invalidation bug: before this
    fix, _remember() fired once per step, so a 2-step run (recall then
    chat) persisted the turn TWICE -- once with the raw tool-result
    dump as the "answer" and once with the real final answer, with the
    user's message duplicated alongside each. Both the duplication and
    the raw-dump entry are wrong: memory.json must end up with exactly
    one exchange per turn, and it must be the clean final answer, not
    an intermediate tool result.
    """
    import forge.memory as memory_mod
    from forge import rag
    from forge.tools import memory as memory_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(memory_mod, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    monkeypatch.setitem(TOOLS, "memory", memory_tool.run)

    conn = rag.get_connection()
    rag.remember(conn, kind="fact", content="Possède un Steam Deck", project=None)
    conn.close()

    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "tool": "memory",
                    "content": json.dumps({"action": "recall", "query": "matériel"}),
                    "done": False,
                }
            )
        return json.dumps({"tool": "chat", "content": "Tu as un Steam Deck !"})

    monkeypatch.setattr(orch_mod, "call_llm", fake_llm)
    result = Orchestrator(max_steps=2).run("Tu peux me lister mon matériel ?")

    assert result.ok
    history = memory_mod.get_history()
    assert len(history) == 2  # exactly one user/assistant exchange, not two
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Tu peux me lister mon matériel ?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Tu as un Steam Deck !"


def test_multi_step_run_keeps_history_untouched_by_step_context(monkeypatch):
    """
    The actual cache fix: `history` passed to the router must stay
    exactly what was loaded from memory.json for the whole run --
    intermediate tool results go through step_context instead, never
    mutating history. This is what keeps the router prompt's history
    block byte-identical between the last call of one turn and the
    first call of the next, so llama-server can reuse the KV cache for
    it instead of invalidating it every turn.
    """
    calls = {"n": 0}
    seen_history_by_call = []

    def fake_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "code", "content": "print(1)", "done": False})
        return json.dumps({"tool": "chat", "content": "done"})

    import forge.router as router_mod

    real_build = router_mod.build_router_prompt

    def spying_build(user_input, history=None, step_context=None, **kw):
        seen_history_by_call.append(history)
        return real_build(user_input, history=history, step_context=step_context, **kw)

    monkeypatch.setattr(orch_mod, "call_llm", fake_llm)
    monkeypatch.setattr(orch_mod, "build_router_prompt", spying_build)

    Orchestrator(max_steps=3).run("write code and explain it")

    assert len(seen_history_by_call) == 2
    # Same history object contents on both calls of this run -- the
    # step 1 tool result never got folded into it.
    assert seen_history_by_call[0] == seen_history_by_call[1]


def test_provider_failure_is_reported_not_raised(monkeypatch):
    from forge.errors import ProviderError

    def boom(prompt):
        raise ProviderError("backend down")

    monkeypatch.setattr(orch_mod, "call_llm", boom)
    result = Orchestrator().run("hello")
    assert not result.ok
    assert "backend down" in result.error


def test_max_steps_is_respected_even_at_zero(monkeypatch):
    monkeypatch.setattr(
        orch_mod, "call_llm", lambda prompt: '{"tool":"chat","content":"x"}'
    )
    import pytest

    from forge.errors import LoopGuardError

    with pytest.raises(LoopGuardError):
        Orchestrator(max_steps=0).run("hello")


def test_missing_done_field_stays_single_step(monkeypatch):
    """
    Backward compatibility: JSON without a "done" field (every model
    and every test predating this feature) must still stop after
    exactly one step, even when max_steps allows more.
    """
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "chat", "content": "hi"}),
    )
    result = Orchestrator(max_steps=5).run("hello")
    assert result.ok
    assert result.steps == 1
    assert result.output == "hi"


def test_done_false_continues_to_a_second_step(monkeypatch):
    """
    An explicit "done": false must make the orchestrator route again,
    using the previous tool's result as context, up to max_steps.
    """
    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"tool": "code", "content": "print(1)", "done": False})
        return json.dumps({"tool": "chat", "content": "here is the explanation"})

    monkeypatch.setattr(orch_mod, "call_llm", fake_llm)
    result = Orchestrator(max_steps=3).run("write code and explain it")

    assert result.ok
    assert result.steps == 2
    assert result.tool == "chat"
    assert result.output == "here is the explanation"


def test_memory_recall_done_false_rephrases_naturally(monkeypatch, tmp_path):
    """
    Reproduces the real usage pattern this was built for: a recall
    step (raw bullet-list output, not a sentence) chains into a second
    step via "done": false, and the router's second call sees the
    folded-in result and can phrase a natural reply -- the whole
    reason the memory tool's recall example sets done:false.
    """
    from forge import rag
    from forge.tools import memory as memory_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    monkeypatch.setitem(TOOLS, "memory", memory_tool.run)

    conn = rag.get_connection()
    rag.remember(conn, kind="fact", content="Possède un Steam Deck", project=None)
    conn.close()

    calls = {"n": 0}

    def fake_llm(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps(
                {
                    "tool": "memory",
                    "content": json.dumps({"action": "recall", "query": "matériel"}),
                    "done": False,
                }
            )
        assert "Possède un Steam Deck" in prompt  # folded result reached step 2
        return json.dumps({"tool": "chat", "content": "Tu as un Steam Deck !"})

    monkeypatch.setattr(orch_mod, "call_llm", fake_llm)
    result = Orchestrator(max_steps=2).run("Tu peux me lister mon matériel ?")

    assert result.ok
    assert result.steps == 2
    assert result.tool == "chat"
    assert result.output == "Tu as un Steam Deck !"


def test_memory_recall_done_false_has_no_effect_at_max_steps_one(monkeypatch, tmp_path):
    """The documented gotcha: with MAX_STEPS=1 (the default), done:false
    is silently ignored and the raw list is what the user sees."""
    from forge import rag
    from forge.tools import memory as memory_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    monkeypatch.setitem(TOOLS, "memory", memory_tool.run)

    conn = rag.get_connection()
    rag.remember(conn, kind="fact", content="Possède un Steam Deck", project=None)
    conn.close()

    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {
                "tool": "memory",
                "content": json.dumps({"action": "recall", "query": "matériel"}),
                "done": False,
            }
        ),
    )
    result = Orchestrator(max_steps=1).run("Tu peux me lister mon matériel ?")

    assert result.ok
    assert result.steps == 1
    assert result.output == "- [fact] Possède un Steam Deck"


def test_memory_repeated_call_now_hard_fails_like_any_other_tool(monkeypatch, tmp_path):
    """
    This used to be memory's own loop-guard fallback test: the router
    routed a natural-language recall as memory:recall + "done": false,
    which sometimes looped, so memory got the same graceful-degrade
    fallback as web_search. That chaining is gone (see
    graphs/recall.py's docstring) -- "recall" is a single deterministic
    call the router only ever makes once, and plain "memory" (remember
    only, now) has no reason to legitimately repeat. A repeat is back
    to being a genuine signal worth surfacing, like every tool except
    web_search always was.
    """
    from forge import rag
    from forge.tools import memory as memory_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    monkeypatch.setitem(TOOLS, "memory", memory_tool.run)

    remember_call = json.dumps(
        {
            "tool": "memory",
            "content": json.dumps(
                {"action": "remember", "kind": "fact", "content": "Possède un Deck"}
            ),
            # "done": false forces a second step so the repeat actually
            # happens within one run -- remember normally completes in
            # one step (done defaults True), so this is a deliberately
            # artificial trigger, only to exercise the loop guard path.
            "done": False,
        }
    )
    monkeypatch.setattr(orch_mod, "call_llm", lambda p: remember_call)

    result = Orchestrator(max_steps=2).run("mémorise deux fois la même chose")

    assert not result.ok
    assert "Stopped" in result.output


def test_recall_is_a_single_dispatch_that_never_reaches_the_loop_guard(
    monkeypatch, tmp_path
):
    """
    Companion to graphs/test_recall.py, at the orchestrator boundary:
    the router makes exactly ONE decision ("recall"), and the graph
    handles recall -> synthesize internally without ever routing back
    through the orchestrator's loop. There is nothing here to trip
    the loop guard on, unlike the old memory:recall + "done": false
    dance this replaces.
    """
    from forge import rag
    from forge.graphs import recall as recall_graph
    from forge.tools import recall as recall_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    monkeypatch.setitem(TOOLS, "recall", recall_tool.run)

    conn = rag.get_connection()
    rag.remember(conn, kind="fact", content="Possède un Steam Deck", project=None)
    conn.close()

    route_call = json.dumps(
        {"tool": "recall", "content": "Tu peux me lister mon matériel ?"}
    )
    calls = []

    def fake_call_llm(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return route_call  # the orchestrator's routing call
        return "Tu as un Steam Deck."  # the graph's internal synthesis call

    monkeypatch.setattr(orch_mod, "call_llm", fake_call_llm)
    monkeypatch.setattr(recall_graph, "call_llm", fake_call_llm)

    result = Orchestrator(max_steps=2).run("Tu peux me lister mon matériel ?")

    assert result.ok
    assert result.tool == "recall"
    assert result.output == "Tu as un Steam Deck."
    assert len(result.trace) == 1  # one routing decision, not two


def test_web_search_repeated_call_degrades_gracefully_instead_of_erroring(
    monkeypatch,
):
    """
    Reproduces a real failure from live testing: the small local model
    repeated the identical web_search call on the second step instead
    of following the steering hint (switching to chat or web_fetch),
    tripping the generic loop guard -- confirmed with two different
    hint phrasings (prose-only, then an explicit worked JSON example)
    and with prompt caching disabled to rule out a cache-reuse bug, so
    this is a genuine small-model self-correction limit, not a prompt
    or infra problem. Same fallback as memory's recall: degrade to the
    already-successful previous result instead of surfacing the
    internal "Stopped: the router tried to repeat the same step."
    message.
    """
    from forge.tools.registry import TOOLS

    monkeypatch.setitem(
        TOOLS, "web_search", lambda content: "Search results for 'Zig':\n1. ..."
    )

    search_call = json.dumps(
        {"tool": "web_search", "content": "langage Zig", "done": False}
    )
    # Always returns the identical call -> the loop guard trips on step 2.
    monkeypatch.setattr(orch_mod, "call_llm", lambda p: search_call)

    result = Orchestrator(max_steps=2).run("Cherche des infos sur Zig")

    assert result.ok
    assert result.error is None
    assert result.tool == "web_search"
    assert result.output == "Search results for 'Zig':\n1. ..."
    assert "Stopped" not in result.output


def test_non_memory_non_web_search_tool_repeat_still_hard_fails(monkeypatch):
    """The safety net above is scoped to memory and web_search only --
    every other tool must keep failing loud on a repeated call, since
    that's a genuine bug signal there, not a known small-model quirk."""
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda p: json.dumps({"tool": "code", "content": "print(1)", "done": False}),
    )
    result = Orchestrator(max_steps=3).run("write code")

    assert not result.ok
    assert result.error is not None
    assert "repeated call" in result.error
    assert "Stopped" in result.output


def test_done_false_stops_at_max_steps_without_crashing(monkeypatch):
    """
    If the router keeps asking for more steps (done: false) beyond
    max_steps, the run must still return the last good result instead
    of raising — max_steps is a ceiling on continued looping, not an
    additional failure mode on top of it.
    """

    def fake_llm(prompt):
        # Vary content so the loop-guard (seen_calls) never triggers.
        n = fake_llm.n
        fake_llm.n += 1
        return json.dumps({"tool": "chat", "content": f"step {n}", "done": False})

    fake_llm.n = 0
    monkeypatch.setattr(orch_mod, "call_llm", fake_llm)
    result = Orchestrator(max_steps=2).run("keep going forever")

    assert result.ok
    assert result.steps == 2
    assert result.output == "step 1"


def test_failed_step_stops_the_loop_even_if_not_done(monkeypatch):
    """
    A tool failure must never be treated as a safe base to route
    again from, regardless of the "done" flag.
    """
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {"tool": "unknown_tool", "content": "x", "done": False}
        ),
    )
    result = Orchestrator(max_steps=3).run("hello")
    # unknown tool falls back to chat in the parser, which always
    # succeeds — so this exercises the "done" respected on success path.
    assert result.tool == "chat"
