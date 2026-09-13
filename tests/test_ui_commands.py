"""
What the web UI's `!` commands actually do, run rather than read.

Every message starting with `!` is handled in the browser and never
reaches the server (`sendChat` -> `runUiCommand`), so this table is the
whole of what those commands are. A command missing from it does not
degrade to the router: it answers "Commande inconnue dans l'interface
web".

That is exactly how `!pair` shipped broken. It was intercepted in
orchestrator.run, tested at that level, tested through the REPL, and
unreachable from the one interface it was written for -- because the
browser answered first and nothing executed this table to notice.

Skipped where deno is absent, like test_ui_rendering.py.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import forge.api as api_mod
from tests.js.extract_renderer import commands_module

INDEX = Path(api_mod.__file__).parent / "static" / "index.html"

pytestmark = pytest.mark.skipif(
    shutil.which("deno") is None, reason="no deno on this machine"
)


@pytest.fixture(scope="module")
def run_command(tmp_path_factory):
    """
    Run one UI command against a stubbed apiFetch.

    Returns (result, requests): what the command handed back to
    runUiCommand, and every request it made on the way.
    """
    work = tmp_path_factory.mktemp("ui_commands")
    (work / "commands.js").write_text(commands_module(INDEX), encoding="utf-8")

    def _run(name: str, args: list[str] | None = None, response: dict | None = None):
        spec = {
            "name": name,
            "args": args or [],
            "response": response or {"status": 200, "body": {}},
        }
        (work / "input.json").write_text(json.dumps(spec), encoding="utf-8")
        (work / "run.js").write_text(
            """
import { commands } from "./commands.js";
const spec = JSON.parse(await Deno.readTextFile("input.json"));
const requests = [];
const apiFetch = async (url, opts = {}) => {
  requests.push({ url, method: opts.method || "GET", body: opts.body || null });
  return {
    ok: spec.response.status < 400,
    status: spec.response.status,
    json: async () => spec.response.body,
  };
};
const table = commands(apiFetch, () => true);
const result = await table[spec.name].run(spec.args);
console.log(JSON.stringify({ result, requests }));
""",
            encoding="utf-8",
        )
        out = subprocess.run(
            ["deno", "run", "--quiet", "--allow-read", "run.js"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert out.returncode == 0, out.stderr
        payload = json.loads(out.stdout)
        return payload["result"], payload["requests"]

    return _run


class TestPairReachesTheServer:
    def test_it_is_a_known_command_at_all(self, run_command):
        """
        The bug, stated as a test. Absent from this table, `!pair`
        answered "Commande inconnue dans l'interface web" and never
        left the browser -- while passing every server-side test.
        """
        _, requests = run_command(
            "!pair", response={"status": 200, "body": {"output": "![QR](data:)"}}
        )
        assert requests, "!pair made no request at all"

    def test_it_posts_the_command_to_chat(self, run_command):
        _, requests = run_command(
            "!pair", response={"status": 200, "body": {"output": "![QR](data:)"}}
        )

        assert len(requests) == 1
        assert requests[0]["url"] == "/chat"
        assert requests[0]["method"] == "POST"
        assert json.loads(requests[0]["body"]) == {"message": "!pair"}

    def test_the_server_answer_is_shown_verbatim(self, run_command):
        """
        The QR is markdown built server-side. Rewriting any of it here
        would be a second copy of the refusal messages and the TTL
        wording, drifting from the first.
        """
        markdown = "**Appairage**\n\n![QR code](data:image/png;base64,AAA)"
        result, _ = run_command(
            "!pair", response={"status": 200, "body": {"output": markdown}}
        )

        assert result["reply"] == markdown

    def test_it_never_reloads_the_history(self, run_command):
        """
        changed:false, always. The turn is deliberately not persisted
        server-side, so a reload would erase the QR code it just drew
        -- and this is the one reply that cannot be fetched again.
        """
        result, _ = run_command(
            "!pair", response={"status": 200, "body": {"output": "![QR](data:)"}}
        )
        assert result["changed"] is False

    def test_a_server_error_is_reported_not_swallowed(self, run_command):
        result, _ = run_command("!pair", response={"status": 500, "body": {}})

        assert "500" in result["reply"]
        assert result["changed"] is False


def test_every_command_the_help_lists_can_run(run_command):
    """
    Guards the guard. `runUiCommand` prints this table as the list of
    what works here, so an entry that is only a help string is a
    promise the interface cannot keep.
    """
    result, _ = run_command("!memory", response={"status": 200, "body": {"total": 0}})
    assert "reply" in result
