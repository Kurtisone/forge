"""
Dispatchable wrapper around forge.graphs.sysadmin.

Same reasoning as tools/review.py and tools/research.py: Forge's UI is
a single conversational page, zero tabs -- this is what lets a plain
chat message ("le service searxng plante", "mon deck rame ce matin")
trigger graphs.sysadmin.run() through the normal router -> dispatch
path.

Unlike review's file_path (which needs workspace confinement because
it names something on disk this process can touch), target_hint here
names a *running* unit or container -- confinement instead happens one
layer down, inside graphs.sysadmin.collect_node, by requiring the name
to appear verbatim in that same run's own discovery output. There is
nothing to confine at this wrapper boundary because there is no path
here yet from target_hint straight to a subprocess call.

Read-only, always: this tool (like graphs/sysadmin.py underneath it)
can never restart, stop, or otherwise mutate anything -- it discovers,
reads logs, and proposes. Applying any fix stays a human action, the
same posture already chosen for tools/git.py.

Interface: run(content: str) -> str
  content is a JSON string:
    {"target_hint": "...", "question": "..."}
  Both fields are optional. With no target_hint, the graph falls back
  to kernel logs (journalctl -k).

To activate: ENABLED_TOOLS=chat,code,sysadmin in .env.local
"""

import json

from forge.graphs.sysadmin import run as sysadmin_run
from forge.tool_payload import loads_payload


def run(content: str) -> str:
    content = content.strip()
    if not content:
        # No content is still a valid request -- "regarde ce qui ne va
        # pas" with no target named -- fall through to the graph with
        # both fields empty rather than rejecting it.
        payload = {}
    else:
        try:
            payload = loads_payload(content, "sysadmin")
        except (json.JSONDecodeError, TypeError):
            return '[error] sysadmin content must be JSON: {"target_hint": "...", "question": "..."}'

    if not isinstance(payload, dict):
        return '[error] sysadmin content must be a JSON object: {"target_hint": "...", "question": "..."}'

    target_hint = payload.get("target_hint") or None
    question = payload.get("question") or None

    return sysadmin_run(target_hint, question)
