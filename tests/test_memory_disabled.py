"""
What MEMORY_ENABLED=false actually does.

conftest.py pins the flag ON for every other test, because five of
them wrote a turn and read it back and had no business depending on a
developer's .env for that. The price of pinning it is that nothing
would exercise the OFF branch any more, and a flag nobody tests is a
flag that quietly stops working -- so this file switches it off
deliberately and states what "off" means:

  - the router prompt is built with no history at all (_recall)
  - a successful turn is not written to memory.json (_finish)

Both importers are patched here for the same reason conftest does it:
api.py and orchestrator.py copy the value at import time, so setting
it on forge.config alone would change nothing they read. That is
precisely the trap this file exists to keep visible.
"""

import json

import forge.api as api_mod
import forge.config as cfg
import forge.memory as memory_mod
import forge.orchestrator as orch_mod
from forge.orchestrator import Orchestrator


def _disable_memory(monkeypatch):
    for module in (cfg, api_mod, orch_mod):
        monkeypatch.setattr(module, "MEMORY_ENABLED", False)


def test_recall_returns_no_history_when_memory_is_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_mod, "MEMORY_FILE", str(tmp_path / "memory.json"))
    memory_mod.add_message("user", "une question d'avant")
    assert memory_mod.get_history()  # the file really does hold something

    _disable_memory(monkeypatch)

    assert Orchestrator()._recall() == []


def test_a_successful_turn_is_not_persisted_when_memory_is_disabled(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(memory_mod, "MEMORY_FILE", str(tmp_path / "memory.json"))
    _disable_memory(monkeypatch)
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt, grammar=None: json.dumps(
            {"tool": "chat", "content": "hi there"}
        ),
    )

    result = Orchestrator().run("hello")

    assert result.ok
    assert "hi there" in result.output
    assert memory_mod.get_history() == []


def test_the_flag_is_read_from_config_by_both_importers():
    """
    Not a behaviour test: a shape test. If either module ever stops
    importing the value and reads forge.config.MEMORY_ENABLED at call
    time instead, the two fixtures above become unnecessary -- and
    this test failing is the signal to simplify them, rather than
    leaving two modules patched forever for a reason that expired.
    """
    assert hasattr(api_mod, "MEMORY_ENABLED")
    assert hasattr(orch_mod, "MEMORY_ENABLED")
