"""
Tests for forge.tools.files, forge.graphs.review, and forge.cli replay.
"""

import json

import forge.config as cfg
import forge.graphs.review as review_mod
import forge.tools.files as files_mod
from forge.graph import Graph
from forge.graphs.review import build as build_review

# -------------------------------------------------------------------
# files tool
# -------------------------------------------------------------------


def test_files_write_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(
        json.dumps({"action": "write", "path": "hello.txt", "content": "Bonjour !"})
    )
    assert "[ok]" in r

    r = files_mod.run(json.dumps({"action": "read", "path": "hello.txt"}))
    assert r == "Bonjour !"


def test_files_write_over_existing_returns_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(
        json.dumps({"action": "write", "path": "f.txt", "content": "line1\nline2\n"})
    )
    r = files_mod.run(
        json.dumps({"action": "write", "path": "f.txt", "content": "line1\nline3\n"})
    )

    assert "```diff" in r
    assert "-line2" in r
    assert "+line3" in r
    # the file on disk must actually be updated, not just diffed
    assert (tmp_path / "f.txt").read_text() == "line1\nline3\n"


def test_files_write_diff_keeps_lines_separate_without_trailing_newline(
    tmp_path, monkeypatch
):
    """
    Regression test: a file with no trailing newline (e.g. a
    single-line script -- the real case this was caught on) made
    unified_diff's last line come out with no line ending either,
    which then joined directly onto the next line with no separator
    ("-old+new" glued together in the rendered diff). Each diff line
    must end up on its own line regardless of the source file's
    trailing-newline situation.
    """
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(
        json.dumps(
            {"action": "write", "path": "hello.py", "content": "print('Hello World')"}
        )
    )
    r = files_mod.run(
        json.dumps(
            {"action": "write", "path": "hello.py", "content": "print('Bienvenue')"}
        )
    )

    assert "-print('Hello World')\n" in r
    assert "+print('Bienvenue')" in r
    assert "World')+print" not in r  # the exact glued-together bug


def test_files_write_identical_content_reports_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(json.dumps({"action": "write", "path": "f.txt", "content": "same"}))
    r = files_mod.run(
        json.dumps({"action": "write", "path": "f.txt", "content": "same"})
    )

    assert "inchangé" in r
    assert "```diff" not in r


def test_files_write_new_path_has_no_diff(tmp_path, monkeypatch):
    """A brand-new file has nothing to diff against -- plain confirmation."""
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(
        json.dumps({"action": "write", "path": "new.txt", "content": "content"})
    )

    assert "[ok] written" in r
    assert "```diff" not in r


def test_files_write_over_oversized_existing_file_skips_diff(tmp_path, monkeypatch):
    """Diffing is skipped (not attempted) against a huge existing file,
    consistent with the read tool's own size guard -- the write itself
    must still succeed."""
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "_MAX_READ_BYTES", 10)

    (tmp_path / "big.txt").write_text("x" * 100)
    r = files_mod.run(
        json.dumps({"action": "write", "path": "big.txt", "content": "short"})
    )

    assert "[ok] written" in r
    assert "```diff" not in r
    assert (tmp_path / "big.txt").read_text() == "short"


def test_files_list(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(json.dumps({"action": "write", "path": "a.txt", "content": "x"}))
    r = files_mod.run(json.dumps({"action": "list", "path": "."}))
    assert "a.txt" in r


def test_files_traversal_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(json.dumps({"action": "read", "path": "../../etc/passwd"}))
    assert "[error]" in r
    assert "permission" in r.lower()


def test_files_leading_slash_means_workspace_root(tmp_path, monkeypatch):
    """Regression test: a router-emitted '/hello.go' used to be
    rejected as 'escaping the workspace' (a pathlib join with an
    absolute right-hand side silently discards the workspace prefix,
    so the traversal check correctly but unhelpfully rejected it) even
    though 'hello.go' worked fine and both should mean the same file
    at the workspace root -- observed live: the router chose 'files'
    for the same request with and without a leading slash and only
    one of the two worked."""
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(
        json.dumps({"action": "write", "path": "hello.go", "content": "package main"})
    )
    r = files_mod.run(json.dumps({"action": "read", "path": "/hello.go"}))
    assert r == "package main"


def test_files_read_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(json.dumps({"action": "read", "path": "missing.txt"}))
    assert "[error]" in r


def test_files_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run("not json")
    assert "[error]" in r


def test_files_unknown_action(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(json.dumps({"action": "delete", "path": "x"}))
    assert "[error]" in r


# -------------------------------------------------------------------
# review graph
# -------------------------------------------------------------------


def test_review_reads_file_and_calls_llm(tmp_path, monkeypatch):
    test_file = tmp_path / "sample.py"
    test_file.write_text("def add(a, b):\n    return a + b\n")

    monkeypatch.setattr(review_mod, "call_llm", lambda p: "Simple and readable.")

    state = build_review().run(
        str(test_file),
        initial_context={"file_path": str(test_file), "question": "Improve?"},
    )
    assert state.ok
    assert "Simple" in state.final_output
    assert state.steps_taken == 2
    assert "file_content" in state.context


def test_review_missing_file_is_graceful(monkeypatch):
    monkeypatch.setattr(review_mod, "call_llm", lambda p: "ok")

    state = build_review().run(
        "/nonexistent/file.py",
        initial_context={"file_path": "/nonexistent/file.py"},
    )
    assert state.ok  # error node recovered
    assert "[error]" in state.final_output


def test_review_provider_failure(tmp_path, monkeypatch):
    from forge.errors import ProviderError

    test_file = tmp_path / "f.py"
    test_file.write_text("x = 1")

    monkeypatch.setattr(
        review_mod, "call_llm", lambda p: (_ for _ in ()).throw(ProviderError("down"))
    )
    state = build_review().run(
        str(test_file),
        initial_context={"file_path": str(test_file)},
    )
    assert not state.ok
    assert "LLM unavailable" in state.final_output


def test_review_shows_degenerate_json_echo_instead_of_silently_truncating(
    tmp_path, monkeypatch
):
    """
    Regression test for the exact bug hit in production use: a small
    model heavily fine-tuned on the router's JSON habit answered a
    review prompt (explicitly asking for plain text) with a
    degenerate JSON echo instead of real analysis. The old
    implementation reused router.parser.parse_router_output on this
    output, which dutifully "succeeded" at extracting {"content"} as
    if it were the real answer -- silently producing an 8-character
    result that was just the reviewed file's name, no visible sign
    anything had gone wrong.

    The fix shows the raw (cleaned) text as-is instead of unwrapping
    it as JSON, so a degenerate response is visibly wrong rather than
    silently truncated to something that happens to look valid.
    """
    src_file = tmp_path / "hello.go"
    src_file.write_text("package main")

    degenerate_raw = '{"tool":"chat","content":"hello.go"}'
    monkeypatch.setattr(review_mod, "call_llm", lambda p: degenerate_raw)

    state = build_review().run(
        str(src_file),
        initial_context={"file_path": str(src_file)},
    )

    assert state.ok
    # The full degenerate JSON is visible, not silently unwrapped to
    # just "hello.go" -- a human reading this immediately sees
    # something went wrong, instead of mistaking it for a real answer.
    assert state.final_output == degenerate_raw
    assert state.final_output != "hello.go"


def test_review_unwraps_substantive_json_wrapped_answer(tmp_path, monkeypatch):
    """
    Regression test for the second real occurrence: the model kept
    wrapping its answer in router-style JSON even after the prompt
    fix (previous commit), but this time the wrapped 'content' was a
    genuine multi-sentence review, not a degenerate echo. Unlike the
    degenerate case, this should be unwrapped to clean prose -- the
    threshold (>= 8 words or >= 40 chars) is what tells the two
    apart.
    """
    src_file = tmp_path / "hello.go"
    src_file.write_text("package main")

    substantive_wrapped = (
        '{ "tool": "chat", "content": "The code is correct and follows Go '
        "best practices. It is a minimal, idiomatic implementation of the "
        "Hello World program. There are no performance, security, or "
        'correctness issues to address." }'
    )
    monkeypatch.setattr(review_mod, "call_llm", lambda p: substantive_wrapped)

    state = build_review().run(
        str(src_file),
        initial_context={"file_path": str(src_file)},
    )

    assert state.final_output.startswith("The code is correct")
    assert '"tool"' not in state.final_output


def test_review_does_not_unwrap_degenerate_short_json_content(tmp_path, monkeypatch):
    """Companion to the unwrap test above: a short, word-poor 'content'
    (the original bug's exact shape) must NOT be unwrapped -- it's not
    trustworthy as a real answer, so the raw JSON stays visible."""
    src_file = tmp_path / "hello.go"
    src_file.write_text("package main")

    degenerate_wrapped = '{"tool":"chat","content":"hello.go"}'
    monkeypatch.setattr(review_mod, "call_llm", lambda p: degenerate_wrapped)

    state = build_review().run(
        str(src_file),
        initial_context={"file_path": str(src_file)},
    )

    assert state.final_output == degenerate_wrapped


def test_review_prompt_includes_json_warning_and_worked_example():
    """
    Locks in the fix for the degenerate-JSON-echo bug at the prompt
    level, not just the response-cleaning level: the prompt must show
    the model a concrete GOOD ANSWER and explicitly labeled
    NEVER DO THIS shapes, not just an abstract 'no JSON' instruction
    -- an instruction alone was already in the prompt before this bug
    was hit in production and did not prevent it.
    """
    prompt = review_mod._REVIEW_PROMPT.format(
        filename="f.py", question="Q?", content="x=1", test_section=""
    )
    assert "GOOD ANSWER" in prompt
    assert "NEVER DO THIS" in prompt
    assert '{"tool":"chat"' in prompt
    assert '{"tool":"code"' in prompt


def test_review_strips_think_blocks_from_response(tmp_path, monkeypatch):
    src_file = tmp_path / "f.py"
    src_file.write_text("x = 1")

    monkeypatch.setattr(
        review_mod,
        "call_llm",
        lambda p: "<think>reasoning about the code...</think>Looks fine overall.",
    )

    state = build_review().run(
        str(src_file),
        initial_context={"file_path": str(src_file)},
    )

    assert state.final_output == "Looks fine overall."


def test_review_long_response_is_not_capped_at_400_chars(tmp_path, monkeypatch):
    """
    Regression test: routing plain-text review answers through
    router.parser.parse_router_output meant every review response
    over 400 chars (_MAX_FALLBACK_CHARS, sized for router tool
    decisions, not analysis prose) was silently truncated -- a
    perfectly legitimate multi-paragraph review would get cut off
    mid-sentence. The dedicated review cleaner has its own, much
    higher ceiling (_MAX_REVIEW_OUTPUT_CHARS = 4000).
    """
    src_file = tmp_path / "f.py"
    src_file.write_text("x = 1")

    long_answer = "This is a detailed review point. " * 20  # ~700 chars
    assert len(long_answer) > 400
    monkeypatch.setattr(review_mod, "call_llm", lambda p: long_answer)

    state = build_review().run(
        str(src_file),
        initial_context={"file_path": str(src_file)},
    )

    assert state.final_output == long_answer.strip()


def test_review_with_test_path_runs_tests_first(tmp_path, monkeypatch):
    import forge.tools.test as test_tool_mod

    src_file = tmp_path / "add.py"
    src_file.write_text("def add(a, b):\n    return a + b\n")

    monkeypatch.setattr(test_tool_mod, "run", lambda cmd: "2 passed")
    captured_prompt = {}

    def fake_call_llm(prompt):
        captured_prompt["prompt"] = prompt
        return "Looks correct, tests pass."

    monkeypatch.setattr(review_mod, "call_llm", fake_call_llm)

    state = build_review().run(
        str(src_file),
        initial_context={
            "file_path": str(src_file),
            "test_path": "tests/test_add.py",
        },
    )

    assert state.ok
    assert "Looks correct" in state.final_output
    # 3 steps this time: read_file -> run_tests -> llm_review
    assert state.steps_taken == 3
    assert "2 passed" in captured_prompt["prompt"]


def test_review_without_test_path_skips_run_tests_node(tmp_path, monkeypatch):
    """No test_path -- behaves exactly like the pre-merge 2-step graph,
    run_tests is never visited."""
    src_file = tmp_path / "f.py"
    src_file.write_text("x = 1")

    monkeypatch.setattr(review_mod, "call_llm", lambda p: "ok")

    state = build_review().run(
        str(src_file),
        initial_context={"file_path": str(src_file)},
    )

    assert state.ok
    assert state.steps_taken == 2
    node_names = [t.decision_tool for t in state.trace]
    assert "run_tests" not in node_names


# -------------------------------------------------------------------
# initial_context flows through graph
# -------------------------------------------------------------------


def test_initial_context_reaches_node():
    def ctx_node(state):
        state.final_output = state.context.get("greeting", "missing")
        return state

    g = Graph("ctx_test")
    g.add_node("n", ctx_node)
    s = g.run("hello", initial_context={"greeting": "Bonjour !"})
    assert s.final_output == "Bonjour !"
