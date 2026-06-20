import json
from pathlib import Path

MEMORY_FILE = Path("data/memory.json")
MAX_HISTORY = 20


# ----------------------------
# LOAD MEMORY SAFE
# ----------------------------
def load_memory():
    if not MEMORY_FILE.exists():
        return {"history": [], "facts": []}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {"history": [], "facts": []}

    if "facts" not in data:
        data["facts"] = []

    if "history" not in data:
        data["history"] = []

    return data


# ----------------------------
# SAVE MEMORY
# ----------------------------
def save_memory(memory):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            memory,
            f,
            indent=2,
            ensure_ascii=False
        )

# ----------------------------
# HISTORY
# ----------------------------
def get_history():
    memory = load_memory()
    return memory.get("history", [])

def safe_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")

def add_message(role, content):
    memory = load_memory()

    history = memory.get("history", [])

    history.append({
        "role": role,
        "content": safe_text(content)
    })

    history = history[-MAX_HISTORY:]

    memory["history"] = history

    save_memory(memory)


# ----------------------------
# FACTS
# ----------------------------
def get_facts():
    memory = load_memory()
    return memory.get("facts", [])


def add_fact(key, value):
    memory = load_memory()

    facts = memory.get("facts", [])

    # overwrite strict
    facts = [f for f in facts if f["key"] != key]

    facts.append({
        "key": key,
        "value": value
    })

    memory["facts"] = facts

    save_memory(memory)


def update_fact(key, value):
    add_fact(key, value)
