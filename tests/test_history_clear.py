"""
Tests for clearing the history from something other than the REPL.

The debt: typing `!clear` in the web UI sent the literal string to
/chat. The router picked `chat`, the model produced a confident
sentence about having cleared the context, and BOTH turns stayed in
the history the command was supposed to empty. The REPL has had
`!clear` since v3.9; the UI never grew an equivalent, and nothing
anywhere said the command did not exist there.

Two halves, and only the server one is testable without a browser:
the endpoint, and the UI intercepting `!` before it can reach the
router. The UI half is pinned by reading the file, which is weak but
strictly better than the nothing that let this ship.
"""

import pathlib

import pytest
from fastapi.testclient import TestClient

from forge import api as api_mod
from forge import ratelimit

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "src" / "forge" / "static" / "index.html"


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_clearing_removes_the_history_and_says_how_much(monkeypatch, tmp_path):
    from forge import memory

    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    memory.save_memory(memory._fresh())
    for i in range(4):
        memory.add_message("user", f"message {i}")

    removed = memory.clear_history()

    assert removed == 4, (
        "clear_history used to return None -- fine for a REPL printing a "
        "fixed line, useless for a caller that needs success from no-op"
    )
    assert memory.load_memory()["history"] == []
    assert memory.clear_history() == 0


def test_the_endpoint_exists_and_is_behind_the_token(monkeypatch):
    routes = {r.path for r in api_mod.app.routes}
    assert "/history/clear" in routes, (
        "the web UI has no other way to clear; without this route the "
        "command goes to the router and comes back invented"
    )

    monkeypatch.setattr(api_mod, "API_TOKEN", "secret")
    client = TestClient(api_mod.app)

    assert client.post("/history/clear").status_code in (401, 403)


def test_the_ui_never_forwards_a_bang_command_to_the_router():
    """
    The regression itself. sendChat must divert anything starting with
    '!' before the /chat call, including commands it does not know --
    the router has no notion of REPL commands, so forwarding one always
    produces an answer about something that did not happen.
    """
    source = INDEX.read_text()

    assert "text.startsWith('!')" in source
    assert "runUiCommand" in source
    assert "Commande inconnue" in source, (
        "an unrecognised ! command must be refused locally, not sent on"
    )


def test_the_ui_confirms_before_clearing():
    """Destructive, includes pinned messages, and not undoable."""
    source = INDEX.read_text()

    assert "confirm(" in source
    assert "/history/clear" in source
