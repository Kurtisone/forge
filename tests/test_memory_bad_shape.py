"""
Valid JSON of the wrong TYPE must not crash a turn.

memory.py's contract is that memory is best effort: an unreadable
file is logged and the run continues on an empty one. That held for
unparseable text and not for parseable text of the wrong shape --
`[]` walked past the JSONDecodeError handler and died on
.setdefault(), surfacing as a 500 on GET /drawer with the UI simply
not loading.

Not hypothetical: with no reset endpoint, the documented way to start
a fresh conversation is to echo a JSON literal into this file by
hand, and one bracket typed instead of a brace produces exactly this.
"""

import json

import pytest

from forge import memory


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '["bonjour"]',
        '"just a string"',
        "42",
        "null",
        "true",
    ],
)
def test_wrong_toplevel_type_starts_fresh(tmp_path, monkeypatch, raw):
    path = tmp_path / "memory.json"
    path.write_text(raw, encoding="utf-8")
    monkeypatch.setattr(memory, "MEMORY_FILE", str(path))

    assert memory.load_memory() == {"history": [], "facts": [], "next_id": 1}


def test_a_fresh_memory_is_not_shared_between_calls(tmp_path, monkeypatch):
    """
    The reason _fresh() builds its value instead of copying a
    constant: callers append to these lists.
    """
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "nope.json"))

    first = memory.load_memory()
    first["history"].append({"id": 1, "role": "user", "content": "x"})

    assert memory.load_memory()["history"] == []


@pytest.mark.parametrize("bad", ["{}", "[]", '"x"', "3"])
def test_wrong_history_type_is_reset(tmp_path, monkeypatch, bad):
    path = tmp_path / "memory.json"
    path.write_text(
        f'{{"history": {bad}, "facts": [], "next_id": 1}}', encoding="utf-8"
    )
    monkeypatch.setattr(memory, "MEMORY_FILE", str(path))

    assert memory.load_memory()["history"] == []


def test_non_object_history_entries_are_dropped_and_the_rest_kept(
    tmp_path, monkeypatch
):
    """
    Repair, not discard: four bad entries are not a reason to throw
    away four hundred good ones.
    """
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            {
                "history": [
                    {"id": 1, "role": "user", "content": "gardé", "pinned": False},
                    "bonjour",
                    None,
                    {"id": 2, "role": "assistant", "content": "aussi", "pinned": False},
                ],
                "facts": [],
                "next_id": 3,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(memory, "MEMORY_FILE", str(path))

    history = memory.load_memory()["history"]
    assert [m["content"] for m in history] == ["gardé", "aussi"]


@pytest.mark.parametrize("bad", ["0", "-1", '"7"', "true", "null", "1.5"])
def test_bad_next_id_never_hands_out_an_id_already_in_use(tmp_path, monkeypatch, bad):
    """
    ids are how pinning and deletion address a message. Reusing one is
    worse than the crash this file is about.
    """
    path = tmp_path / "memory.json"
    path.write_text(
        '{"history": [{"id": 1, "role": "user", "content": "a", "pinned": false},'
        '{"id": 7, "role": "user", "content": "b", "pinned": false}],'
        f'"facts": [], "next_id": {bad}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(memory, "MEMORY_FILE", str(path))

    assert memory.load_memory()["next_id"] == 8


def test_a_repaired_file_still_serves_the_endpoints(tmp_path, monkeypatch):
    """
    The symptom that started this: GET /drawer returning 500. Checked
    through the public helpers rather than load_memory alone, since
    that is where the AttributeError actually surfaced.
    """
    path = tmp_path / "memory.json"
    path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(memory, "MEMORY_FILE", str(path))

    assert memory.get_history() == []
    memory.add_message("user", "après réparation")
    assert [m["content"] for m in memory.get_history()] == ["après réparation"]
