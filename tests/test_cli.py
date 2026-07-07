"""
Tests for forge.cli: the `forge review` and `forge replay` commands
documented in the README's Usage section since v3.1, with 0% test
coverage until now.

No network / no real LLM: forge.graphs.review.run is monkeypatched
for review, and trace.read_last is monkeypatched for replay.
"""

import sys

import pytest

import forge.cli as cli_mod


# ── forge review ─────────────────────────────────────────────────────

def test_review_with_no_file_prints_usage_and_returns_1(capsys):
    code = cli_mod._cmd_review([])
    err = capsys.readouterr().err
    assert code == 1
    assert "Usage: forge review" in err


def test_review_calls_graph_with_default_question(monkeypatch, capsys):
    captured = {}

    def fake_run(file_path, question="Que peut-on améliorer ?"):
        captured["file_path"] = file_path
        captured["question"] = question
        return "looks fine"

    monkeypatch.setattr("forge.graphs.review.run", fake_run)
    code = cli_mod._cmd_review(["src/forge/main.py"])

    assert code == 0
    assert captured["file_path"] == "src/forge/main.py"
    assert captured["question"] == "Que peut-on améliorer ?"
    assert "looks fine" in capsys.readouterr().out


def test_review_joins_extra_args_into_the_question(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "forge.graphs.review.run",
        lambda file_path, question="x": captured.setdefault("question", question),
    )
    cli_mod._cmd_review(["file.py", "Any", "security", "issues?"])
    assert captured["question"] == "Any security issues?"


# ── forge replay ──────────────────────────────────────────────────────

def test_replay_with_no_run_id_prints_usage_and_returns_1(capsys):
    code = cli_mod._cmd_replay([])
    err = capsys.readouterr().err
    assert code == 1
    assert "Usage: forge replay" in err


def test_replay_no_match_returns_1(monkeypatch, capsys):
    monkeypatch.setattr("forge.trace.read_last", lambda n=100: [])
    code = cli_mod._cmd_replay(["abc123"])
    err = capsys.readouterr().err
    assert code == 1
    assert "No trace found" in err


def test_replay_prints_matching_trace_details(monkeypatch, capsys):
    fake_trace = {
        "run_id": "abc12345",
        "timestamp": "2026-07-01T00:00:00Z",
        "user_input_preview": "hello",
        "ok": True,
        "total_ms": 42,
        "steps": [
            {
                "router_tool": "chat",
                "duration_ms": 40,
                "tool_ok": True,
                "router_content_preview": "hi there",
                "tool_error": None,
            }
        ],
        "error": None,
    }
    monkeypatch.setattr("forge.trace.read_last", lambda n=100: [fake_trace])
    code = cli_mod._cmd_replay(["abc123"])
    out = capsys.readouterr().out

    assert code == 0
    assert "abc12345" in out
    assert "✓ ok" in out
    assert "chat" in out
    assert "hi there" in out


def test_replay_picks_the_last_of_multiple_prefix_matches(monkeypatch):
    older = {"run_id": "abc111", "steps": []}
    newer = {"run_id": "abc222", "steps": []}
    monkeypatch.setattr("forge.trace.read_last", lambda n=100: [older, newer])

    # read_last returns oldest-first (matches trace.py's own convention);
    # replay must pick the last (most recent) match, not the first.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        cli_mod._cmd_replay(["abc"])
    assert "abc222" in buf.getvalue()
    assert "abc111" not in buf.getvalue()


def test_replay_shows_run_level_error_if_present(monkeypatch, capsys):
    fake_trace = {"run_id": "err001", "steps": [], "error": "provider unavailable"}
    monkeypatch.setattr("forge.trace.read_last", lambda n=100: [fake_trace])
    cli_mod._cmd_replay(["err001"])
    out = capsys.readouterr().out
    assert "provider unavailable" in out


def test_replay_shows_per_step_tool_error(monkeypatch, capsys):
    fake_trace = {
        "run_id": "stp001",
        "steps": [
            {
                "router_tool": "shell",
                "duration_ms": 5,
                "tool_ok": False,
                "router_content_preview": "rm -rf /",
                "tool_error": "command not in allowlist",
            }
        ],
    }
    monkeypatch.setattr("forge.trace.read_last", lambda n=100: [fake_trace])
    cli_mod._cmd_replay(["stp001"])
    out = capsys.readouterr().out
    assert "command not in allowlist" in out
    assert "✗" in out


# ── main() dispatch ───────────────────────────────────────────────────

def test_main_with_no_args_prints_help_and_exits_0(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["forge"])
    with pytest.raises(SystemExit) as exc:
        cli_mod.main()
    assert exc.value.code == 0
    assert "forge review" in capsys.readouterr().out


def test_main_with_help_flag_exits_0(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["forge", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli_mod.main()
    assert exc.value.code == 0


def test_main_with_unknown_command_exits_1(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["forge", "not-a-real-command"])
    with pytest.raises(SystemExit) as exc:
        cli_mod.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Unknown command" in err
    assert "review" in err and "replay" in err


def test_main_dispatches_to_review(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["forge", "review", "f.py"])
    monkeypatch.setattr(cli_mod, "_cmd_review", lambda args: 0)
    with pytest.raises(SystemExit) as exc:
        cli_mod.main()
    assert exc.value.code == 0
