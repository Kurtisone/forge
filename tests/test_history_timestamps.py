"""
Tests for the timestamp carried by each history entry.

The debt: an entry was `{id, role, content, pinned}` and nothing more.
Ids are monotonic, but they say nothing about elapsed time, so a
client reloading the whole thread -- the Android app does exactly that
on every launch -- had no way to tell this morning's turns from last
week's, and could group the conversation by nothing at all.

The field is written going forward only. Entries already on disk
cannot be given a truthful date, and inventing one would be worse than
admitting there is none, so it has to survive being absent.
"""

import time

import pytest
from fastapi.testclient import TestClient

from forge import api as api_mod
from forge import ratelimit


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_a_new_turn_is_stamped_with_the_current_time(monkeypatch, tmp_path):
    from forge import memory

    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    memory.save_memory(memory._fresh())

    before = time.time()
    memory.add_message("user", "quelle heure est-il")
    after = time.time()

    entry = memory.get_history()[0]
    assert before <= entry["ts"] <= after


def test_entries_written_before_the_field_existed_stay_readable(monkeypatch, tmp_path):
    """
    The shape that predates this field must keep working: no ts, and no
    backfilled one either. A date that was never recorded cannot be
    invented, so it stays absent and the client decides what to show.
    """
    from forge import memory

    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    memory.save_memory(
        {
            "history": [
                {"id": 1, "role": "user", "content": "ancien", "pinned": False}
            ],
            "facts": [],
            "next_id": 2,
        }
    )

    entry = memory.get_history()[0]
    assert "ts" not in entry


def test_the_api_reports_the_stamp_and_its_absence(monkeypatch, tmp_path):
    from forge import memory

    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(api_mod, "API_TOKEN", "")
    monkeypatch.setattr(api_mod, "API_ALLOW_UNAUTHENTICATED", True)
    memory.save_memory(
        {
            "history": [
                {"id": 1, "role": "user", "content": "ancien", "pinned": False}
            ],
            "facts": [],
            "next_id": 2,
        }
    )
    memory.add_message("assistant", "récent")

    client = TestClient(api_mod.app)
    body = client.get("/history").json()

    assert body[0]["ts"] is None, "une entrée sans date ne doit pas en recevoir une"
    assert isinstance(body[1]["ts"], float)
