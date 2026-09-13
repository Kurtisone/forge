"""
`!pair` as a whole turn, through the orchestrator.

Two properties that forge/pairing.py cannot hold by itself, because
both are decided by what run() does around it: the turn costs no LLM
call, and it is not written down anywhere.

The second one is the load-bearing half. A QR code is a live
credential plus a kilobyte of base64, and _finish persists a turn to
three places that each make keeping it wrong -- memory.json (rendered
back by GET /history on every page load), the rolling history (prefix
to every router prompt afterwards) and the vector store (reachable by
a recall months later).
"""

import json

import pytest

from forge import memory, pairing
from forge import orchestrator as orch_mod

_URL = "http://10.8.0.1:8000"
_BEARER = "durable-bearer-token-that-must-not-leak"


@pytest.fixture(autouse=True)
def _configured(monkeypatch):
    monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", _URL)
    monkeypatch.setattr(pairing, "API_TOKEN", _BEARER)
    pairing.reset()
    yield
    pairing.reset()


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    """
    Any LLM call at all fails the test that triggers it.

    Same guarantee delegation.py's interception carries and for the
    same reason: this runs above _recall(), which can trigger
    compaction, which calls the model. Showing a QR code should not
    cost a prefill on an APU.
    """

    def _explode(*args, **kwargs):
        raise AssertionError("an intercepted turn must not reach the model")

    monkeypatch.setattr(orch_mod, "call_llm", _explode)


def test_the_turn_answers_with_the_qr_code():
    result = orch_mod.Orchestrator().run("!pair")

    assert result.ok
    assert result.tool == "pair"
    assert "](data:image/png;base64," in result.output


def test_the_turn_is_not_written_to_memory():
    """
    The whole reason this interception carries remember=False. A
    pairing code is worth five minutes; nothing about it should
    outlive the screen it was drawn on.
    """
    orch_mod.Orchestrator().run("!pair")
    assert memory.get_history() == []


def test_the_bearer_token_reaches_neither_the_reply_nor_the_store():
    result = orch_mod.Orchestrator().run("!pair")

    assert _BEARER not in result.output
    assert _BEARER not in json.dumps(memory.get_history())


def test_a_refusal_is_not_written_down_either(monkeypatch):
    """
    A misconfigured !pair answers with an error, which is still not a
    turn worth keeping -- and it is the path most likely to be
    repeated a dozen times while the address gets fixed.
    """
    monkeypatch.setattr(pairing, "FORGE_PUBLIC_URL", "")
    result = orch_mod.Orchestrator().run("!pair")

    assert "FORGE_PUBLIC_URL" in result.output
    assert memory.get_history() == []


def test_pair_wins_over_a_job_waiting_on_an_answer():
    """
    Ordering, stated as a test because it is invisible otherwise: a
    waiting job takes the next message as the answer to its question,
    so intercepting !pair after delegation would file "!pair" as an
    objective and leave the user with no QR code and a corrupted spec.
    """
    from forge import jobs

    job = jobs.create({})
    jobs.transition(job.id, jobs.AWAITING_USER, pending_field="objective")

    result = orch_mod.Orchestrator().run("!pair")

    assert "](data:image/png;base64," in result.output
    assert jobs.get(job.id).spec.get("objective") is None


def test_an_ordinary_message_still_reaches_the_router(monkeypatch):
    """
    The common case: pairing.intercept returns None and run() proceeds
    exactly as it did before this existed.
    """
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": "chat", "content": "bonjour"}),
    )
    result = orch_mod.Orchestrator().run("dis-moi bonjour")

    assert result.tool == "chat"
    assert memory.get_history(), "an ordinary turn is still remembered"
