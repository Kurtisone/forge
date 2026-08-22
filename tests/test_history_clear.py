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


def test_a_command_reply_is_appended_after_any_refresh():
    """
    Bug in the first version of this feature, reported live: `!memory`
    did nothing visible and the typed command vanished from the chat.

    The reply WAS produced -- GET /memory?limit=100 returned 200 in the
    server log -- and then loadHistory() rebuilt the chat from the
    server, erasing both the echoed command and its answer, neither of
    which the server knows about. `!clear` survived by accident (the
    history is empty afterwards, so the confirm popup was the only
    visible effect) and `!truc` survived because the unknown-command
    branch returns before the refresh. `!memory`, whose entire output
    is local, disappeared without a trace.

    From the outside that is indistinguishable from a command that
    does nothing at all.
    """
    source = INDEX.read_text()
    body = source[source.index("async function runUiCommand") :]
    body = body[: body.index("async function sendChat")]

    refresh_at = body.index("loadHistory()")
    append_at = body.index("appendMsg(null, reply")

    assert append_at > refresh_at, (
        "the reply must be appended AFTER the history refresh, or the refresh erases it"
    )


def test_a_command_reports_whether_it_changed_anything():
    """
    A read-only command has nothing to re-read, and reloading for it is
    what created the bug above.

    The first fix declared this statically, per command, and was wrong
    for the case that matters most. Reported live: !truc, then !memory,
    then !clear answered "no" at the confirm -- and the refresh ran
    anyway, erasing the two earlier replies. Cancelling something must
    not have more visible effect than doing it.

    So `changed` comes back from run() with the result. A cancelled
    clear reports false; a compaction that condensed nothing reports
    false too.
    """
    source = INDEX.read_text()

    assert "changesHistory: true" not in source, (
        "a static flag cannot know that a confirm() was cancelled"
    )
    assert "result.changed === true" in source
    assert "changed: false" in source


def test_cancelling_a_clear_changes_nothing():
    """The exact reported sequence, read out of the source."""
    source = INDEX.read_text()
    clear_block = source[source.index("'!clear':") :]
    clear_block = clear_block[: clear_block.index("'!compact':")]

    cancel_at = clear_block.index("confirm(")
    returned = clear_block[cancel_at : clear_block.index("apiFetch")]

    assert "changed: false" in returned, (
        "the cancelled branch must report changed:false, or the refresh "
        "wipes every earlier local reply"
    )
