"""
Persistent conversation memory: short rolling history + key/value facts.

Storage is a single JSON file (data/memory.json by default, see
MEMORY_FILE in config.py). This is intentionally simple: Forge is a
single-user, single-process local runtime, so there is no
concurrency to manage, and a JSON file is far easier to inspect and
debug ("cat data/memory.json") than a database. Revisit this if
Forge ever needs concurrent writers or queries beyond "last N
messages" / "lookup by key" -- SQLite would be the natural upgrade
at that point, not before.
"""

import json
from pathlib import Path

from forge import compaction
from forge.config import MEMORY_FILE, MEMORY_MAX_HISTORY
from forge.logger import log

_DEFAULT = {"history": [], "facts": [], "next_id": 1}


def _path() -> Path:
    return Path(MEMORY_FILE)


def load_memory() -> dict:
    path = _path()
    if not path.exists():
        return {"history": [], "facts": [], "next_id": 1}

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        # Empty file (e.g. freshly created by a volume mount) is not
        # corrupted, just empty -- nothing to warn about.
        return {"history": [], "facts": [], "next_id": 1}

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("memory file unreadable (%s), starting fresh: %s", path, e)
        return {"history": [], "facts": [], "next_id": 1}

    data.setdefault("history", [])
    data.setdefault("facts", [])
    data.setdefault("next_id", 1)
    if _migrate_history(data):
        save_memory(data)
    return data


def _migrate_history(data: dict) -> bool:
    """
    Backfill 'id' and 'pinned' on history entries written before v3.9
    (which didn't have them). Runs on every load, but only writes back
    to disk the first time it actually finds something to migrate.
    Without this, GET /history's response model rejects any entry
    missing 'id', which fails the whole request for anyone with
    pre-v3.9 conversation history already on disk.
    """
    changed = False
    for m in data["history"]:
        if "id" not in m:
            m["id"] = data["next_id"]
            data["next_id"] += 1
            changed = True
        if "pinned" not in m:
            m["pinned"] = False
            changed = True
    return changed


def save_memory(memory: dict) -> None:
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
    except OSError as e:
        # Memory is a convenience feature: a write failure must never
        # crash a conversation turn.
        log.error("failed to write memory file %s: %s", path, e)


def safe_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")


# ----------------------------
# HISTORY
# ----------------------------
def get_history() -> list[dict]:
    return load_memory().get("history", [])


def _new_entry(memory: dict, role: str, content: str) -> dict:
    entry = {
        "id": memory["next_id"],
        "role": role,
        "content": safe_text(content),
        "pinned": False,
    }
    memory["next_id"] += 1
    return entry


def _apply_retention(history: list[dict]) -> list[dict]:
    """
    Runs after every turn. First tries compaction (v3.9): pinned
    messages are always kept, older non-pinned ones are replaced by a
    single summary once COMPACTION_THRESHOLD is crossed, instead of
    just being dropped. MEMORY_MAX_HISTORY below stays as a hard-cap
    safety net -- drop-oldest, pinned exempt -- for when compaction is
    disabled or its strategy fails; history must never grow unbounded.
    """
    try:
        history = compaction.maybe_compact(history)
    except compaction.CompactionError as e:
        log.warning("compaction failed, falling back to hard cap: %s", e)

    if len(history) <= MEMORY_MAX_HISTORY:
        return history

    pinned = [m for m in history if m.get("pinned")]
    unpinned = [m for m in history if not m.get("pinned")]
    budget = max(MEMORY_MAX_HISTORY - len(pinned), 0)
    return [*pinned, *unpinned[-budget:]] if budget else pinned


def add_message(role: str, content: str) -> None:
    memory = load_memory()
    history = memory.get("history", [])

    history.append(_new_entry(memory, role, content))
    memory["history"] = _apply_retention(history)

    save_memory(memory)


def add_exchange(user_content: str, assistant_content: str) -> None:
    """
    Persist one user/assistant turn in a single read-modify-write,
    instead of calling add_message() twice (which would read and
    rewrite the file twice for what is logically one turn).
    """
    memory = load_memory()
    history = memory.get("history", [])

    history.append(_new_entry(memory, "user", user_content))
    history.append(_new_entry(memory, "assistant", assistant_content))
    memory["history"] = _apply_retention(history)

    save_memory(memory)


def clear_history() -> None:
    """
    Wipe the rolling conversation history (but keep facts).
    Used by the !clear REPL command. Pinned messages are cleared too:
    this command is an explicit reset, not a compaction pass.
    """
    memory = load_memory()
    memory["history"] = []
    save_memory(memory)
    log.info("conversation history cleared")


def compact_now() -> int:
    """
    Force a compaction pass regardless of COMPACTION_THRESHOLD. Used
    by the manual /compact endpoint and the !compact REPL command.
    Returns the number of messages removed from the visible history
    (still recoverable via RAG under the rag_pointer strategy).
    """
    memory = load_memory()
    history = memory.get("history", [])
    before = len(history)

    memory["history"] = compaction.maybe_compact(history, force=True)
    save_memory(memory)

    return before - len(memory["history"])


# ----------------------------
# DRAWER (pinned messages, v3.9)
# ----------------------------
def pin_message(message_id: int) -> bool:
    memory = load_memory()
    for m in memory.get("history", []):
        if m.get("id") == message_id:
            m["pinned"] = True
            save_memory(memory)
            return True
    return False


def unpin_message(message_id: int) -> bool:
    memory = load_memory()
    for m in memory.get("history", []):
        if m.get("id") == message_id:
            m["pinned"] = False
            save_memory(memory)
            return True
    return False


def get_pinned() -> list[dict]:
    return [m for m in get_history() if m.get("pinned")]


# ----------------------------
# FACTS
# ----------------------------
def get_facts() -> list[dict]:
    return load_memory().get("facts", [])


def add_fact(key: str, value: str) -> None:
    memory = load_memory()
    facts = [f for f in memory.get("facts", []) if f["key"] != key]
    facts.append({"key": key, "value": value})
    memory["facts"] = facts
    save_memory(memory)


def update_fact(key: str, value: str) -> None:
    add_fact(key, value)
