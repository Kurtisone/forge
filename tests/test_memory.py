"""
Unit tests for forge.memory. Uses a temp file (monkeypatching the
module-level MEMORY_FILE constant) so nothing touches the real
data/memory.json.
"""

import forge.memory as memory


def test_add_and_get_history(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))

    memory.add_message("user", "hello")
    memory.add_message("assistant", "hi")

    history = memory.get_history()
    assert history == [
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
