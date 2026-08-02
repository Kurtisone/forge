"""
Tests for the router becoming tool-aware (v3.5): the prompt only
describes tools that are actually enabled+loaded, and the parser only
accepts a router-picked tool if it's in that same set.

Before this: files/shell/git were reachable only via an explicit
Graph (POST /run), never from a normal chat turn, even with
ENABLED_TOOLS listing them -- the router's own prompt and validation
hardcoded exactly {"chat", "code"}. Nothing about ENABLED_TOOLS itself
changes here: a tool still has to be explicitly opted into that list
to be reachable either way. What changes is that the router can now
actually offer an already-opted-in tool during a conversation.
"""

import forge.tools.registry as registry_mod
from forge.router.parser import _validate_json_obj, parse_router_output
from forge.router.prompt import build_router_prompt

# ── parser: dynamic tool validation ─────────────────────────────────


def test_unlisted_tool_falls_back_to_chat(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code"])
    decision = _validate_json_obj({"tool": "shell", "content": "ls"}, "raw")
    assert decision.tool == "chat"


def test_enabled_tool_is_accepted(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "available_tools", lambda: ["chat", "code", "shell"]
    )
    decision = _validate_json_obj({"tool": "shell", "content": "ls -la"}, "raw")
    assert decision.tool == "shell"
    assert decision.content == "ls -la"


def test_chat_and_code_always_valid_even_if_registry_excludes_them(monkeypatch):
    """_VALID_TOOLS is a floor: a misconfigured ENABLED_TOOLS that
    somehow excludes chat/code must not make the router itself unable
    to fall back to chat."""
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["shell"])
    decision = _validate_json_obj({"tool": "code", "content": "print(1)"}, "raw")
    assert decision.tool == "code"


def test_parse_router_output_end_to_end_with_files_enabled(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "available_tools", lambda: ["chat", "code", "files"]
    )
    raw = '{"tool":"files","content":"{\\"action\\":\\"read\\",\\"path\\":\\"x.py\\"}"}'
    decision = parse_router_output(raw)
    assert decision.tool == "files"


def test_parse_router_output_end_to_end_with_files_disabled(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code"])
    raw = '{"tool":"files","content":"{\\"action\\":\\"read\\",\\"path\\":\\"x.py\\"}"}'
    decision = parse_router_output(raw)
    assert decision.tool == "chat"


def test_parse_router_output_end_to_end_with_memory_enabled(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "available_tools", lambda: ["chat", "code", "memory"]
    )
    raw = '{"tool":"memory","content":"{\\"action\\":\\"recall\\",\\"query\\":\\"podman\\"}"}'
    decision = parse_router_output(raw)
    assert decision.tool == "memory"


def test_parse_router_output_end_to_end_with_memory_disabled(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code"])
    raw = '{"tool":"memory","content":"{\\"action\\":\\"recall\\",\\"query\\":\\"podman\\"}"}'
    decision = parse_router_output(raw)
    assert decision.tool == "chat"


# ── prompt: dynamic tool descriptions ────────────────────────────────


def test_prompt_with_only_chat_and_code_omits_other_tools():
    prompt = build_router_prompt("hi", available_tools=["chat", "code"])
    assert '"chat" or "code"' in prompt
    assert "shell" not in prompt
    assert "files" not in prompt
    assert "git" not in prompt


def test_prompt_includes_shell_when_enabled():
    prompt = build_router_prompt("run ls", available_tools=["chat", "code", "shell"])
    assert '"shell"' in prompt
    assert "single shell command" in prompt
    assert "ls -la" in prompt  # the shell example


def test_prompt_includes_files_description_when_enabled():
    prompt = build_router_prompt(
        "read a file", available_tools=["chat", "code", "files"]
    )
    assert "action" in prompt
    assert '"read"' in prompt or "read" in prompt


def test_prompt_includes_files_write_example():
    """
    Regression test: the files tool used to have only a 'read' worked
    example, never 'write' -- a small local model asked to create a
    file would answer with a code block as plain chat text instead of
    actually persisting it, since it had never seen the write shape,
    only read about it in the description. Both actions must be
    demonstrated, not just described.
    """
    prompt = build_router_prompt(
        "crée un fichier", available_tools=["chat", "code", "files"]
    )
    assert '"action\\":\\"write\\"' in prompt


def test_prompt_files_write_example_json_is_well_formed():
    import json
    import re

    prompt = build_router_prompt(
        "crée un fichier", available_tools=["chat", "code", "files"]
    )
    match = re.search(
        r'\{"tool":"files","content":"\{\\"action\\":\\"write\\".*?\}"\}',
        prompt,
    )
    assert match, "files write example not found in prompt"
    outer = json.loads(match.group(0))
    assert outer["tool"] == "files"
    inner = json.loads(outer["content"])
    assert inner["action"] == "write"
    assert inner["path"]
    assert inner["content"]


def test_prompt_includes_files_read_for_edit_example():
    """
    Regression test: editing an existing file ("remplace X par Y dans
    hello.go") used to have no worked example at all -- the model
    would answer from memory/guesswork (observed live: several such
    requests in a row all returned the ORIGINAL unmodified content,
    never touching the real file). The read-with-done:false example
    teaches it to fetch the real content first.
    """
    prompt = build_router_prompt(
        "modifie ce fichier", available_tools=["chat", "code", "files"]
    )
    assert '"action\\":\\"read\\"' in prompt
    assert '"done":false' in prompt


def test_prompt_steers_toward_files_write_after_a_files_read():
    """The other half of the read-then-write pattern: after seeing its
    own [files] read result, the model must be pushed to write the
    modified file back, not answer in chat or read again."""
    prompt = build_router_prompt(
        "remplace Hello par Bienvenue",
        history=[{"role": "user", "content": "remplace Hello par Bienvenue"}],
        step_context=[
            {"role": "assistant", "content": "[files] package main\n\nfunc main() {}"}
        ],
        available_tools=["chat", "code", "files"],
    )
    assert 'Do NOT call "action":"read" again' in prompt
    assert '"tool":"files"' in prompt.split("CURRENT, real content")[-1]


def test_prompt_files_read_hint_gives_full_content_more_room():
    """A files:read result must not be squashed to the same 120-char
    cap as ordinary history/tool-result summaries -- the model is
    about to be asked to reproduce it in full with one change applied,
    and a 120-char excerpt would guarantee a truncated rewrite."""
    long_content = "line\n" * 100  # 500 chars, well past the 120-char cap
    prompt = build_router_prompt(
        "remplace X par Y",
        step_context=[{"role": "assistant", "content": f"[files] {long_content}"}],
        available_tools=["chat", "code", "files"],
    )
    assert long_content.strip() in prompt


def test_prompt_omits_files_read_hint_after_a_write_result():
    """A write confirmation/diff must not trigger the read-then-write
    hint -- only an actual read result should."""
    prompt = build_router_prompt(
        "ok",
        step_context=[
            {"role": "assistant", "content": "[files] [ok] written 12 bytes to f.txt"}
        ],
        available_tools=["chat", "code", "files"],
    )
    assert 'Do NOT call "action":"read" again' not in prompt


def test_prompt_includes_git_description_when_enabled():
    prompt = build_router_prompt("git log", available_tools=["chat", "code", "git"])
    assert "git subcommand" in prompt
    assert "Read-only" in prompt


def test_prompt_includes_memory_description_when_enabled():
    prompt = build_router_prompt(
        "remember this decision", available_tools=["chat", "code", "memory"]
    )
    assert '"memory"' in prompt
    assert "remember" in prompt
    assert "recall" in prompt
    assert "explicitly asks" in prompt


def test_prompt_memory_recall_example_demonstrates_done_false():
    """The recall example must show "done": false -- that's what
    teaches a small local model to chain a rephrasing step instead of
    returning the raw bullet-list search results as the final answer."""
    prompt = build_router_prompt(
        "list my hardware", available_tools=["chat", "code", "memory"]
    )
    assert '"action\\":\\"recall\\"' in prompt
    assert '"done":false' in prompt


def test_prompt_memory_recall_example_json_is_well_formed():
    import json
    import re

    prompt = build_router_prompt(
        "list my hardware", available_tools=["chat", "code", "memory"]
    )
    match = re.search(
        r'\{"tool":"memory","content":"\{\\"action\\":\\"recall\\".*?"done":false\}',
        prompt,
    )
    assert match, "recall example not found in prompt"
    outer = json.loads(match.group(0))
    assert outer["tool"] == "memory"
    assert outer["done"] is False
    inner = json.loads(outer["content"])
    assert inner["action"] == "recall"


def test_prompt_with_only_chat_and_code_omits_memory():
    prompt = build_router_prompt("hi", available_tools=["chat", "code"])
    assert "memory" not in prompt


def test_prompt_defaults_to_registry_when_available_tools_not_passed(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "available_tools", lambda: ["chat", "code", "git"]
    )
    prompt = build_router_prompt("hi")
    assert '"git"' in prompt


def test_prompt_falls_back_to_default_pair_if_registry_returns_empty(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", list)
    prompt = build_router_prompt("hi")
    assert '"chat" or "code"' in prompt


def test_prompt_still_fills_in_user_input_and_history():
    prompt = build_router_prompt(
        "what's next?",
        history=[{"role": "user", "content": "earlier message"}],
        available_tools=["chat", "code"],
    )
    assert "what's next?" in prompt
    assert "earlier message" in prompt


def test_prompt_steers_toward_chat_after_a_memory_result():
    """The real failure this guards against: a small local model asked
    to route again right after its own [memory] tool output tends to
    call memory a second time instead of answering -- this hint pushes
    explicitly toward "tool":"chat" instead."""
    prompt = build_router_prompt(
        "Tu peux me lister mon matériel ?",
        history=[
            {"role": "user", "content": "Tu peux me lister mon matériel ?"},
        ],
        step_context=[
            {"role": "assistant", "content": "[memory] - [fact] Possède un Steam Deck"},
        ],
        available_tools=["chat", "code", "memory"],
    )
    assert "Do NOT call the memory tool again" in prompt
    assert '"tool":"chat"' in prompt.split("already contains the answer")[-1]


def test_prompt_memory_hint_includes_a_concrete_rephrasing_example():
    """Second round of live testing: the abstract "in your own words"
    instruction got the model to stop repeating the memory call, but
    it then just copied the raw bullet line verbatim instead of
    actually rephrasing it. A worked before/after example is what
    small local models actually follow -- assert it's really there,
    not just the abstract rule."""
    prompt = build_router_prompt(
        "Tu peux me lister mon matériel ?",
        history=[
            {"role": "user", "content": "Tu peux me lister mon matériel ?"},
        ],
        step_context=[
            {"role": "assistant", "content": "[memory] - [fact] Possède un Steam Deck"},
        ],
        available_tools=["chat", "code", "memory"],
    )
    assert "do NOT copy the" in prompt
    assert "bullet format verbatim" in prompt
    assert "Tu as un Steam Deck !" in prompt  # the worked example itself


def test_prompt_omits_memory_steering_hint_for_non_memory_step_context():
    prompt = build_router_prompt(
        "what's next?",
        history=[{"role": "user", "content": "run some code"}],
        step_context=[{"role": "assistant", "content": "[code] print(1)"}],
        available_tools=["chat", "code", "memory"],
    )
    assert "Do NOT call the memory tool again" not in prompt


def test_prompt_omits_memory_steering_hint_with_no_step_context():
    prompt = build_router_prompt("hello", available_tools=["chat", "code", "memory"])
    assert "Do NOT call the memory tool again" not in prompt


def test_unknown_tool_without_a_description_gets_generic_wording():
    """A custom tool the operator wrote, with no entry in
    TOOL_DESCRIPTIONS, must still produce a valid (if generic) prompt
    section instead of crashing or silently omitting it."""
    prompt = build_router_prompt(
        "do the thing", available_tools=["chat", "custom_tool"]
    )
    assert '"custom_tool"' in prompt
    assert "content is the input this tool expects" in prompt


# ── parser: is_fallback flag (placeholders must not be remembered) ──


def test_repetition_loop_is_flagged_as_fallback():
    repeated = " ".join(["banana"] * 20)
    decision = parse_router_output(repeated)
    assert decision.is_fallback is True


def test_empty_output_is_flagged_as_fallback():
    decision = parse_router_output("   ")
    assert decision.is_fallback is True


def test_history_block_is_stable_regardless_of_step_context():
    """
    The v3.8 cache fix: the history section of the prompt must be
    byte-identical whether or not step_context is present or what it
    contains -- history is meant to always mirror memory.json exactly,
    so llama-server can reuse the KV cache for that whole prefix
    across turns. Only the text after it (step_context, then the user
    input) should ever differ.
    """
    history = [
        {"role": "user", "content": "Tu peux me lister mon matériel ?"},
        {"role": "assistant", "content": "Tu as un Steam Deck !"},
    ]

    no_step_context = build_router_prompt(
        "next question", history=history, available_tools=["chat", "code", "memory"]
    )
    with_step_context = build_router_prompt(
        "next question",
        history=history,
        step_context=[{"role": "assistant", "content": "[code] print(1)"}],
        available_tools=["chat", "code", "memory"],
    )

    history_block_marker = "Context from earlier in this conversation"
    prefix_no_ctx = no_step_context.split(history_block_marker)[0]
    prefix_with_ctx = with_step_context.split(history_block_marker)[0]
    assert prefix_no_ctx == prefix_with_ctx  # static template unaffected

    # And the history bullets themselves are identical in both, right
    # up to where step_context's own block would start.
    history_and_after_no_ctx = no_step_context.split(history_block_marker)[1]
    history_and_after_with_ctx = with_step_context.split(history_block_marker)[1]
    common_history_text = "they said: Tu peux me lister mon matériel ?"
    assert common_history_text in history_and_after_no_ctx
    assert common_history_text in history_and_after_with_ctx


def test_leaked_prompt_is_flagged_as_fallback():
    decision = parse_router_output(
        "some preamble... NEVER add text outside the JSON, ok?"
    )
    assert decision.is_fallback is True


def test_valid_json_decision_is_not_flagged_as_fallback():
    decision = parse_router_output('{"tool":"chat","content":"a real answer"}')
    assert decision.is_fallback is False


def test_plain_text_answer_is_not_flagged_as_fallback():
    """A genuine (if unstructured) answer extracted via the plain-text
    fallback path is still real content -- only the placeholder-
    generating branches (repetition, leaked prompt, empty) count as
    is_fallback."""
    decision = parse_router_output("Bien sûr, voici la réponse à votre question.")
    assert decision.is_fallback is False
    assert "Bien sûr" in decision.content


def test_prompt_includes_review_description_when_enabled():
    prompt = build_router_prompt(
        "relis ce fichier", available_tools=["chat", "code", "review"]
    )
    assert '"review"' in prompt
    assert "file_path" in prompt


def test_prompt_omits_review_when_not_enabled():
    prompt = build_router_prompt("relis ce fichier", available_tools=["chat", "code"])
    assert '"review"' not in prompt


def test_prompt_includes_review_test_path_example():
    """Regression-style: without a worked example showing test_path,
    a small local model only ever sees file_path and never combines a
    review with running that file's tests, even when explicitly asked
    to."""
    prompt = build_router_prompt(
        "relis ce fichier et ses tests", available_tools=["chat", "code", "review"]
    )
    assert '\\"test_path\\"' in prompt


def test_prompt_review_examples_json_is_well_formed():
    import json

    prompt = build_router_prompt(
        "relis ce fichier", available_tools=["chat", "code", "review"]
    )
    for line in prompt.splitlines():
        if '"tool":"review"' in line and line.strip().startswith("{"):
            outer = json.loads(line.strip().rstrip(","))
            inner = json.loads(outer["content"])
            assert "file_path" in inner


def test_prompt_disambiguates_bare_relis_from_review_with_opinion():
    """
    Regression test for the exact ambiguity hit in production use:
    review's own first worked example used to be a bare "relire X"
    with no request for feedback, which taught the model that the
    verb alone means review -- directly conflicting with files' own
    "relis X" -> read example. Both tools' examples must now anchor
    the same verb to different tools based on whether an opinion is
    requested, not just to the presence of "relis"/"relire".
    """
    prompt = build_router_prompt(
        "relis quelque chose",
        available_tools=["chat", "code", "files", "review"],
    )
    # files' bare-read example must be present, unqualified by any
    # opinion request
    assert '"tool":"files"' in prompt
    assert "Relis hello.go" in prompt

    # review's example must pair the same verb with an explicit
    # request for feedback, not stand alone
    assert "et me donner ton avis" in prompt or "donne ton avis" in prompt


def test_prompt_includes_web_fetch_description_and_examples():
    """
    Regression test for a real gap hit in production use: web_fetch
    had no entry at all in TOOL_DESCRIPTIONS/_TOOL_EXAMPLES, so the
    router fell back to the generic "content is the input this tool
    expects" wording and produced malformed content (an empty/non-URL
    string, observed as "[error] unsupported scheme: ''").
    """
    prompt = build_router_prompt(
        "fetch a url", available_tools=["chat", "code", "web_fetch"]
    )
    assert '"web_fetch"' in prompt
    assert "does NOT search" in prompt
    assert "https://example.com/status" in prompt


def test_prompt_web_fetch_defers_to_web_search_when_both_enabled():
    """
    web_fetch's description must point to web_search for vague/no-URL
    requests once web_search exists -- the old behavior (teaching a
    chat refusal for "actualités en bourse") became actively wrong
    once Forge gained real search capability, since it would steer
    the router away from the tool that can now actually help.
    """
    prompt = build_router_prompt(
        "actualités", available_tools=["chat", "code", "web_fetch", "web_search"]
    )
    assert "web_search" in prompt
    # the vague-news example now lives under web_search, not web_fetch
    assert "actualités bourse" in prompt


def test_prompt_omits_web_fetch_when_not_enabled():
    prompt = build_router_prompt("fetch a url", available_tools=["chat", "code"])
    assert '"web_fetch"' not in prompt


def test_prompt_includes_web_search_description_and_examples():
    prompt = build_router_prompt(
        "cherche quelque chose",
        available_tools=["chat", "code", "web_search"],
    )
    assert '"web_search"' in prompt
    assert "not a URL" in prompt
    assert "langage de programmation Zig" in prompt
    assert "actualités bourse" in prompt


def test_prompt_web_search_examples_use_done_false():
    """
    Regression test for a real bug hit in production use: without
    "done":false on the initial web_search call, the orchestrator
    treated the search itself as the complete answer and returned the
    raw results list verbatim to the user instead of ever reaching a
    synthesis/fetch step. Both worked examples must demonstrate this,
    same as memory's "recall" action already does.
    """
    prompt = build_router_prompt(
        "cherche quelque chose",
        available_tools=["chat", "code", "web_search"],
    )
    web_search_block = prompt[prompt.find("Examples:") :]
    for line in web_search_block.splitlines():
        if '"tool":"web_search"' in line:
            assert '"done":false' in line, f"missing done:false in: {line}"


def test_prompt_omits_web_search_when_not_enabled():
    prompt = build_router_prompt(
        "cherche quelque chose", available_tools=["chat", "code"]
    )
    assert '"web_search"' not in prompt


def test_prompt_steers_toward_chat_or_fetch_after_a_web_search_result():
    step_context = [
        {
            "role": "assistant",
            "content": (
                "[web_search] Search results for 'Zig': "
                "1. Zig homepage\n   https://ziglang.org\n   A general-purpose language."
            ),
        }
    ]
    prompt = build_router_prompt(
        "cherche Zig",
        available_tools=["chat", "code", "web_search", "web_fetch"],
        step_context=step_context,
    )
    assert "do NOT call web_search again" in prompt
    assert '"tool":"web_fetch"' in prompt.split("Result from a tool")[-1]


def test_prompt_omits_web_search_steering_hint_with_no_step_context():
    prompt = build_router_prompt(
        "cherche Zig", available_tools=["chat", "code", "web_search"]
    )
    assert "do NOT call web_search again" not in prompt
