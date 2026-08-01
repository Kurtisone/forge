"""
Unit tests for forge.memory. Uses a temp file (monkeypatching the
module-level MEMORY_FILE constant) so nothing touches the real
data/memory.json.
"""

from forge import memory


def test_add_and_get_history(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))

    memory.add_message("user", "hello")
    memory.add_message("assistant", "hi")

    history = memory.get_history()
    assert [{k: m[k] for k in ("role", "content")} for m in history] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_history_is_capped_at_max(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "MEMORY_MAX_HISTORY", 3)

    for i in range(10):
        memory.add_message("user", f"msg{i}")

    history = memory.get_history()
    assert len(history) == 3
    assert history[-1]["content"] == "msg9"


def test_corrupted_file_does_not_crash(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    path.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(memory, "MEMORY_FILE", str(path))

    assert memory.get_history() == []


def test_facts_overwrite_strict(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))

    memory.add_fact("name", "Kurt")
    memory.add_fact("name", "Kurtis")

    facts = memory.get_facts()
    assert facts == [{"key": "name", "value": "Kurtis"}]


# ----------------------------
# DRAWER / COMPACTION (v3.9)
# ----------------------------


def test_messages_get_stable_incrementing_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))

    memory.add_message("user", "hello")
    memory.add_message("assistant", "hi")

    history = memory.get_history()
    assert [m["id"] for m in history] == [1, 2]
    assert all(m["pinned"] is False for m in history)


def test_pin_and_unpin_message(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    memory.add_message("user", "keep this one")
    memory.add_message("user", "not this one")

    assert memory.pin_message(1) is True
    assert [m["id"] for m in memory.get_pinned()] == [1]

    assert memory.unpin_message(1) is True
    assert memory.get_pinned() == []


def test_pin_unknown_id_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    memory.add_message("user", "hello")

    assert memory.pin_message(999) is False


def test_hard_cap_exempts_pinned_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "MEMORY_MAX_HISTORY", 3)
    monkeypatch.setattr(memory.compaction, "COMPACTION_ENABLED", False)

    memory.add_message("user", "msg0")
    memory.pin_message(1)
    for i in range(1, 10):
        memory.add_message("user", f"msg{i}")

    history = memory.get_history()
    assert any(m["id"] == 1 and m["pinned"] for m in history)
    assert len(history) == 3


def test_compact_now_forces_compaction(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory.compaction, "COMPACTION_ENABLED", False)
    monkeypatch.setattr(memory.compaction, "COMPACTION_KEEP_RECENT", 1)

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(memory.compaction.rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(
        memory.compaction.rag, "remember", lambda conn, kind, content, project: 1
    )

    for i in range(5):
        memory.add_message("user", f"msg{i}")

    removed = memory.compact_now()

    assert removed > 0
    history = memory.get_history()
    assert history[0]["role"] == "system"
