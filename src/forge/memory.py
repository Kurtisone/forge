import json
from pathlib import Path


MEMORY_FILE = Path("data/memory.json")
MAX_HISTORY = 20


def load_memory():
    if not MEMORY_FILE.exists():
        return {"history": []}

    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_memory(memory):
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def get_history():
    memory = load_memory()
    return memory.get("history", [])


def add_message(role, content):
    memory = load_memory()

    history = memory.get("history", [])

    history.append({
        "role": role,
        "content": content
    })

    history = history[-MAX_HISTORY:]

    memory["history"] = history

    save_memory(memory)

def get_facts():
    memory = load_memory()
    return memory.get("facts", [])


def add_fact(key, value):
    memory = load_memory()

    facts = memory.get("facts", [])

    # overwrite si existe déjà
    facts = [f for f in facts if f["key"] != key]

    facts.append({
        "key": key,
        "value": value
    })

    memory["facts"] = facts

    save_memory(memory)
