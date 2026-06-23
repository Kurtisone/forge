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

from forge.config import MEMORY_FILE, MEMORY_MAX_HISTORY
from forge.logger import log

_DEFAULT = {"history": [], "facts": []}


def _path() -> Path:
    return Path(MEMORY_FILE)


def load_memory() -> dict:
    path = _path()
    if not path.exists():
        return {"history": [], "facts": []}

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        # Empty file (e.g. freshly created by a volume mount) is not
        # corrupted, just empty -- nothing to warn about.
        return {"history": [], "facts": []}

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("memory file unreadable (%s), starting fresh: %s", path, e)
        return {"history": [], "facts": []}

    data.setdefault("history", [])
    data.setdefault("facts", [])
    return data


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


def add_message(role: str, content: str) -> None:
    memory = load_memory()
    history = memory.get("history", [])

    history.append({"role": role, "content": safe_text(content)})
    memory["history"] = history[-MEMORY_MAX_HISTORY:]

    save_memory(memory)


def add_exchange(user_content: str, assistant_content: str) -> None:
    """
    Persist one user/assistant turn in a single read-modify-write,
    instead of calling add_message() twice (which would read and
    rewrite the file twice for what is logically one turn).
    """
    memory = load_memory()
    history = memory.get("history", [])

    history.append({"role": "user", "content": safe_text(user_content)})
    history.append({"role": "assistant", "content": safe_text(assistant_content)})
    memory["history"] = history[-MEMORY_MAX_HISTORY:]

    save_memory(memory)


def clear_history() -> None:
    """
    Wipe the rolling conversation history (but keep facts).
    Used by the !clear REPL command.
    """
    memory = load_memory()
    memory["history"] = []
    save_memory(memory)
    log.info("conversation history cleared")


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
