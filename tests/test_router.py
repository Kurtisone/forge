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

import json

import forge.tools.registry as registry_mod
from forge.router.parser import _validate_json_obj, parse_router_output
from forge.router.prompt import build_router_prompt


def _example_line(prompt: str, *fragments: str) -> str:
    """The single worked-example line containing all *fragments*.

    Examples are rendered one per line, so matching on the line is
    enough -- and keeps these tests off the exact escaping, which is
    what they kept breaking on.
    """
    matches = [
        line.strip()
        for line in prompt.splitlines()
        if line.strip().startswith("{") and all(f in line for f in fragments)
    ]
    assert len(matches) == 1, (
        f"expected one example matching {fragments}, got {matches}"
    )
    return matches[0]


# ── parser: dynamic tool validation ─────────────────────────────────


def _noop(content: str) -> str:
    """Stand-in handler: these tests care about the prompt, not dispatch."""
    return content


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
    assert '"action":"write"' in prompt


def test_prompt_write_example_body_is_multi_line():
    """hello.py used to be the only write that ever worked, and only
    by accident: a lone print() has no brace to confuse the parser's
    brace scanner and no newline to escape. The worked example has to
    show the shape that actually failed live, or it keeps teaching the
    one case that was never the problem."""
    prompt = build_router_prompt(
        "crée un fichier", available_tools=["chat", "code", "files"]
    )
    example = _example_line(prompt, '"tool":"files"', '"action":"write"')
    body = json.loads(example)["content"]["content"]
    assert "\n" in body


def test_prompt_files_write_example_json_is_well_formed():
    prompt = build_router_prompt(
        "crée un fichier", available_tools=["chat", "code", "files"]
    )
    outer = json.loads(_example_line(prompt, '"tool":"files"', '"action":"write"'))
    assert outer["tool"] == "files"
    inner = outer["content"]
    # A nested object, NOT a string holding re-encoded JSON: that
    # second level of escaping is exactly what the model failed to
    # produce on multi-line content.
    assert isinstance(inner, dict)
    assert inner["action"] == "write"
    assert inner["path"]
    assert inner["content"]


def test_prompt_teaches_one_step_edit_for_an_existing_file():
    """
    Editing an existing file was taught as read(done:false) -> write,
    which needs the router to chain. It doesn't, reliably: observed
    live on "Remplace Hello World par Bonjour à tous", the run stopped
    after the read and answered with the file's ORIGINAL content --
    the same symptom the read example was originally added to fix, and
    the same non-chaining that already forced deterministic handling
    for web_search and memory:recall.

    "edit" collapses it into one dispatch, and never asks the model to
    reproduce the file it just read.
    """
    prompt = build_router_prompt(
        "modifie ce fichier", available_tools=["chat", "code", "files"]
    )
    example = _example_line(prompt, '"tool":"files"', '"action":"edit"')
    payload = json.loads(example)["content"]
    assert payload["find"] and payload["replace"]
    assert payload["path"]


def test_prompt_does_not_teach_chaining_for_a_plain_edit():
    """The steering hint after a files:read still exists for the cases
    edit can't cover (adding a function, restructuring). What must NOT
    come back is a worked example telling the model that a literal
    replacement starts with a done:false read."""
    prompt = build_router_prompt(
        "modifie ce fichier", available_tools=["chat", "code", "files"]
    )
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and '"tool":"files"' in stripped:
            obj = json.loads(stripped)
            if obj["content"].get("action") == "read":
                assert obj.get("done") is not False, (
                    "a files:read example still teaches done:false chaining"
                )


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
        last_read_path="main.go",
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


def test_prompt_recall_examples_never_use_done_false():
    """
    Regression guard for the same bug research already guards against
    (test_prompt_research_examples_never_use_done_false): recall must
    NEVER be shown chained with "done": false in a worked example --
    that's the exact router-chaining bug graphs/recall.py exists to
    avoid (see its docstring).
    """
    prompt = build_router_prompt(
        "list my hardware", available_tools=["chat", "code", "recall"]
    )
    for line in prompt.splitlines():
        if '"tool":"recall"' in line:
            assert '"done"' not in line, f"recall example must not use done: {line}"


def test_prompt_recall_example_json_is_well_formed():
    import json
    import re

    prompt = build_router_prompt(
        "list my hardware", available_tools=["chat", "code", "recall"]
    )
    match = re.search(r'\{"tool":"recall","content":"[^"]*"\}', prompt)
    assert match, "recall example not found in prompt"
    outer = json.loads(match.group(0))
    assert outer["tool"] == "recall"
    assert isinstance(outer["content"], str) and outer["content"]


def test_prompt_recall_description_points_away_from_memory():
    prompt = build_router_prompt(
        "hi", available_tools=["chat", "code", "memory", "recall"]
    )
    assert '"recall"' in prompt
    assert "single call" in prompt.split('"recall": (')[-1].split('"sysadmin"')[0]


def test_prompt_with_only_chat_and_code_omits_memory():
    prompt = build_router_prompt("hi", available_tools=["chat", "code"])
    assert "memory" not in prompt


def test_prompt_defaults_to_registry_when_available_tools_not_passed(monkeypatch):
    # Pinned on TOOLS rather than on available_tools(): the prompt now
    # resolves through the capability registry, which is a view over
    # TOOLS, so TOOLS is the source of truth both paths agree on.
    monkeypatch.setattr(
        registry_mod, "TOOLS", {"chat": _noop, "code": _noop, "git": _noop}
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


def test_prompt_no_longer_steers_after_a_memory_result():
    """
    A [memory] step_context entry used to trigger a steering hint
    pushing the router back toward "tool":"chat" -- that whole
    mechanism is gone, because the router never routes memory:recall
    with "done":false anymore (see graphs/recall.py's docstring). A
    [memory] result in step_context today can only be a "remember"
    confirmation, which needs no steering at all.
    """
    prompt = build_router_prompt(
        "Tu peux me lister mon matériel ?",
        history=[
            {"role": "user", "content": "Tu peux me lister mon matériel ?"},
        ],
        step_context=[
            {"role": "assistant", "content": "[memory] Remembered (#1)."},
        ],
        available_tools=["chat", "code", "memory"],
    )
    assert "Do NOT call the memory tool again" not in prompt
    assert "already contains the answer" not in prompt


def test_prompt_files_steering_hint_still_works():
    """The files-read steering hint is the one real hint left in this
    block now that memory's is gone -- guard that removing memory's
    branch didn't also break the one still needed."""
    prompt = build_router_prompt(
        "ajoute une fonction",
        history=[{"role": "user", "content": "ajoute une fonction"}],
        step_context=[
            {"role": "assistant", "content": "[files] def f(): pass"},
        ],
        available_tools=["chat", "code", "files"],
        last_read_path="f.py",
    )
    assert '"action":"write"' in prompt
    assert "CURRENT, real content" in prompt


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

    history_block_marker = "Conversation so far."
    prefix_no_ctx = no_step_context.split(history_block_marker)[0]
    prefix_with_ctx = with_step_context.split(history_block_marker)[0]
    assert prefix_no_ctx == prefix_with_ctx  # static template unaffected

    # And the history bullets themselves are identical in both, right
    # up to where step_context's own block would start.
    history_and_after_no_ctx = no_step_context.split(history_block_marker)[1]
    history_and_after_with_ctx = with_step_context.split(history_block_marker)[1]
    common_history_text = "\nUser: Tu peux me lister mon matériel ?\n"
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
    assert '"test_path"' in prompt


def test_prompt_review_examples_json_is_well_formed():
    prompt = build_router_prompt(
        "relis ce fichier", available_tools=["chat", "code", "review"]
    )
    seen = 0
    for line in prompt.splitlines():
        if '"tool":"review"' in line and line.strip().startswith("{"):
            outer = json.loads(line.strip().rstrip(","))
            assert isinstance(outer["content"], dict)
            assert "file_path" in outer["content"]
            seen += 1
    assert seen == 2


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


def test_prompt_web_fetch_defers_to_research_when_available():
    """
    web_fetch's description must point to "research" for vague/no-URL
    requests once it exists -- the old behavior (teaching a chat
    refusal, then later a web_search deferral) became stale once
    Forge gained a reliable single-call research tool. web_search
    chaining into a second router-decided step reliably failed with
    this model, which is exactly why "research" exists.
    """
    prompt = build_router_prompt(
        "actualités", available_tools=["chat", "code", "web_fetch", "research"]
    )
    assert "research" in prompt
    # the vague-news example now lives under research, not web_fetch
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
    assert "langage Zig" in prompt


def test_prompt_includes_research_description_and_examples():
    """
    "research" (search -> fetch -> synthesize as one deterministic
    graph run, see graphs/research.py) must have its own description
    and examples distinct from web_search -- it's the default choice
    for an actual answer/summary about something current, while
    web_search stays for when the user wants links/sources themselves.
    """
    prompt = build_router_prompt(
        "actualité du jeu vidéo",
        available_tools=["chat", "code", "research"],
    )
    assert '"research"' in prompt
    assert "one call" in prompt or "internally" in prompt
    assert "actualité jeu vidéo" in prompt
    assert "actualités bourse" in prompt


def test_prompt_research_examples_never_use_done_false():
    """
    Regression guard for the opposite bug: research must NEVER be
    taught with "done":false, since it's explicitly a single,
    self-contained call (search+fetch+synthesize happen inside the
    graph, not across router steps). Adding done:false here would
    reintroduce the same multi-step chaining reliability problem
    research exists specifically to avoid.
    """
    prompt = build_router_prompt(
        "actualité du jeu vidéo",
        available_tools=["chat", "code", "research"],
    )
    examples_block = prompt[prompt.find("Examples:") :]
    for line in examples_block.splitlines():
        if '"tool":"research"' in line:
            assert '"done"' not in line, f"research example must not use done: {line}"


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
    assert "Do NOT call web_search again" in prompt
    assert '"tool":"chat","content":"<natural answer' in prompt
    assert '"tool":"web_fetch","content":"<that result' in prompt


def test_prompt_omits_web_search_steering_hint_with_no_step_context():
    prompt = build_router_prompt(
        "cherche Zig", available_tools=["chat", "code", "web_search"]
    )
    assert "Do NOT call web_search again" not in prompt


def test_prompt_includes_todays_date():
    """
    Regression test for a real issue observed live: the model
    confused 2025 and 2026 in a research synthesis with no date
    grounding at all, only correcting itself once the user manually
    stated the date in their own prompt. The router prompt must
    always state today's date so this doesn't depend on the user
    remembering to do that themselves.
    """
    from forge.context_info import today_line

    prompt = build_router_prompt("hello", available_tools=["chat", "code"])
    assert today_line() in prompt


def test_prompt_includes_vague_file_reference_instruction_when_history_exists():
    """
    Closes a real unresolved case from v3.9: "analyse le contenu"
    referring implicitly to a file mentioned earlier in the
    conversation (not named again) could make the model answer from
    imagined content instead of ever reading the real file. The
    history block must always carry an instruction to resolve a vague
    later reference against the most recent real file path already
    visible in that same history, rather than fabricating one.
    """
    history = [
        {"role": "user", "content": "Crée un fichier notes.py avec x = 1"},
        {"role": "assistant", "content": "[ok] written 9 bytes to notes.py"},
    ]
    prompt = build_router_prompt(
        "améliore le contenu",
        available_tools=["chat", "code", "files"],
        history=history,
    )
    assert "refers to a file vaguely" in prompt
    assert "notes.py" in prompt  # the real path stayed visible in history


def test_vague_file_reference_instruction_is_gated_on_tools_not_history():
    """
    This instruction used to be emitted only when history was non-empty.
    That gate was wrong twice over.

    It broke the cacheable prefix: a block that appears for the first
    time on turn 2 is an insertion in front of turn 1's text, i.e. one
    guaranteed cache miss per conversation for a fixed piece of static
    guidance.

    And history was never what made it meaningful -- the tool set is.
    Gating on the tool set keeps the promise made at the top of
    router/prompt.py (a tool not opted into via ENABLED_TOOLS is never
    named in the prompt) while staying constant for the lifetime of the
    process.
    """
    with_files = build_router_prompt(
        "améliore le contenu", available_tools=["chat", "code", "files"]
    )
    assert "refers to a file vaguely" in with_files  # no history needed

    without_files = build_router_prompt(
        "améliore le contenu", available_tools=["chat", "code"]
    )
    assert "refers to a file vaguely" not in without_files
    assert "files" not in without_files


def test_delegate_is_described_when_enabled(monkeypatch):
    """
    A tool the prompt never mentions is a tool the router never picks.
    Worth pinning down because adding one is not free: the prompt
    floor is the dominant per-call cost on this box, so "delegate"
    ships with two examples rather than the six it could have had.
    """
    monkeypatch.setattr(
        registry_mod, "TOOLS", {"chat": _noop, "code": _noop, "delegate": _noop}
    )
    prompt = build_router_prompt("délègue la correction du cache KV")
    assert "delegate" in prompt


def test_delegate_is_absent_when_not_enabled(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code"])
    assert "delegate" not in build_router_prompt("délègue la correction du cache KV")
