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

from forge import compaction, tokens
from forge.config import MEMORY_FILE, MEMORY_HARD_CAP_SLACK, MEMORY_MAX_HISTORY
from forge.logger import log


def _path() -> Path:
    return Path(MEMORY_FILE)


def _fresh() -> dict:
    """
    An empty memory. Built each call, never a shared module constant
    copied around: the value holds two lists that callers go on to
    append to, and a shallow copy of a constant would have every
    "fresh" memory in the process sharing the same history.
    """
    return {"history": [], "facts": [], "next_id": 1}


def load_memory() -> dict:
    path = _path()
    if not path.exists():
        return _fresh()

    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        # Empty file (e.g. freshly created by a volume mount) is not
        # corrupted, just empty -- nothing to warn about.
        return _fresh()

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("memory file unreadable (%s), starting fresh: %s", path, e)
        return _fresh()

    # Valid JSON of the WRONG TYPE used to walk straight past the
    # except clause above and die on .setdefault() -- an AttributeError
    # from a module whose entire contract is "memory is best effort,
    # never crash a turn", surfacing as a 500 on GET /drawer with the
    # UI simply not loading.
    #
    # `[]` is not a hypothetical: the documented way to start a fresh
    # conversation is to echo a JSON literal into this file by hand
    # (there is no reset endpoint), and one bracket typed instead of a
    # brace produces exactly this. The recovery is the same as for
    # unparseable JSON -- say so and start fresh -- because a file
    # that isn't an object holds nothing this module knows how to
    # read.
    if not isinstance(data, dict):
        log.warning(
            "memory file %s holds a %s, not an object -- starting fresh",
            path,
            type(data).__name__,
        )
        return _fresh()

    # Same failure one level down, and just as reachable by hand: a
    # history that isn't a list breaks _migrate_history's loop, and a
    # non-dict entry inside it breaks the assignment in that loop.
    # Repaired rather than discarded -- dropping four bad entries is
    # not a reason to throw away four hundred good ones.
    for key, kind in (("history", list), ("facts", list)):
        value = data.setdefault(key, kind())
        if not isinstance(value, kind):
            log.warning(
                "memory file %s: %r is a %s, not a %s -- resetting that field",
                path,
                key,
                type(value).__name__,
                kind.__name__,
            )
            data[key] = kind()

    kept = [entry for entry in data["history"] if isinstance(entry, dict)]
    if len(kept) != len(data["history"]):
        log.warning(
            "memory file %s: dropped %d history entries that were not objects",
            path,
            len(data["history"]) - len(kept),
        )
        data["history"] = kept

    # bool is an int in Python, and `"next_id": true` would go on to
    # hand every entry the id 1.
    next_id = data.setdefault("next_id", 1)
    if not isinstance(next_id, int) or isinstance(next_id, bool) or next_id < 1:
        log.warning(
            "memory file %s: 'next_id' is %r -- recomputing from the history",
            path,
            next_id,
        )
        data["next_id"] = 1

    # An id already in use would be handed out again by _new_entry,
    # and ids are what pinning and deletion address messages by.
    used = [e["id"] for e in data["history"] if isinstance(e.get("id"), int)]
    if used:
        data["next_id"] = max(data["next_id"], max(used) + 1)

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

    When it does fire, it trims to MEMORY_HARD_CAP_SLACK *below* the
    cap rather than to the cap exactly. Landing on the cap put the
    next turn back over it immediately, so one message was evicted
    every turn from then on: a sliding window, shifting the router
    prompt's prefix every turn and forcing llama-server to re-process
    all of it. That is precisely the FIFO/KV-cache conflict v3.8
    diagnosed and raised MEMORY_MAX_HISTORY to avoid -- the safety net
    was quietly reintroducing it. Cutting deeper costs a bit more
    context at once and buys a stable prefix for many turns after.
    """
    # Observation only, and deliberately BEFORE compaction runs: the
    # size that matters is the one that would have been sent, not what
    # is left after eviction. Nothing reads this yet.
    #
    # This is the STORED size. It is not what the history weighs in the
    # router prompt -- _format_history truncates assistant entries to
    # 120 chars, so the two diverge by more than a factor of two on a
    # normal conversation (measured 2026-08-16). The prompt-side figure
    # is logged separately as "router.history_block", and that is the
    # one a token budget should be keyed on. This one says what
    # compaction is evicting; that one says what evicting it buys.
    #
    # Both are logged rather than one, because switching compaction
    # from a message count to a token budget needs constants nobody can
    # pick from first principles, and picking them off the wrong metric
    # is worse than not measuring at all. Same discipline as
    # MEMORY_HARD_CAP_SLACK below, which exists because a threshold
    # chosen without measuring quietly reintroduced the sliding window
    # it was meant to prevent.
    log.event(
        "history.size",
        messages=len(history),
        stored_tokens=tokens.estimate_messages(history),
        pinned=sum(1 for m in history if m.get("pinned")),
    )

    try:
        history = compaction.maybe_compact(history)
    except compaction.CompactionError as e:
        log.warning("compaction failed, falling back to hard cap: %s", e)

    if len(history) <= MEMORY_MAX_HISTORY:
        return history

    pinned = [m for m in history if m.get("pinned")]
    unpinned = [m for m in history if not m.get("pinned")]
    target = max(MEMORY_MAX_HISTORY - MEMORY_HARD_CAP_SLACK, 1)
    budget = max(target - len(pinned), 0)
    kept = [*pinned, *unpinned[-budget:]] if budget else pinned

    log.warning(
        "hard cap: history trimmed from %d to %d messages (%d dropped) -- "
        "compaction is disabled or failing, and the router prompt prefix "
        "just changed, so expect one slow turn",
        len(history),
        len(kept),
        len(history) - len(kept),
    )
    return kept


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
