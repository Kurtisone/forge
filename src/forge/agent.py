import json
from forge.llm import call_llm
from forge.memory import add_message, get_history, update_fact, get_facts


SYSTEM_PROMPT = """
Tu es Forge, un assistant pour développeur.

FACTS est une base de données persistante sur l'utilisateur.

Règles :
- FACTS est la vérité
- Tu ne modifies jamais FACTS directement
- Tu peux proposer des informations, mais pas les appliquer toi-même
- En cas de conflit, FACTS gagnent
"""


# ----------------------------
# MULTI FACT EXTRACTION
# ----------------------------
def extract_facts(user_input: str):
    prompt = f"""
Extract ALL structured facts from the user input.

Return ONLY valid JSON array.

Each item must be:
{{
  "key": "...",
  "value": "..." or ["..."]
}}

Rules:
- Detect multiple facts in one sentence
- If user expresses preferences, use list
- If no fact, return []

Input:
{user_input}
"""

    raw = call_llm(prompt)

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


# ----------------------------
# INTENT CLASSIFIER (simple)
# ----------------------------
def detect_intent(user_input: str):
    prompt = f"""
Classify user intent.

Return ONLY JSON:
{{
  "intent": "chat | query_fact"
}}

User:
{user_input}
"""

    raw = call_llm(prompt)

    try:
        return json.loads(raw)
    except Exception:
        return {"intent": "chat"}


# ----------------------------
# AGENT CORE
# ----------------------------
def run_agent(user_input: str):
    # 1. save user message
    add_message("user", user_input)

    # 2. extract facts (MULTI)
    new_facts = extract_facts(user_input)

    # 3. store facts
    for fact in new_facts:
        key = fact.get("key")
        value = fact.get("value")

        if key and value:
            update_fact(key, value)

    # 4. detect intent
    intent_data = detect_intent(user_input)
    intent = intent_data.get("intent", "chat")

    facts = get_facts()
    history = get_history()

    # 5. build context
    facts_text = ""
    for f in facts:
        facts_text += f"- {f['key']}: {f['value']}\n"

    history_text = ""
    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    # 6. QUERY FACT MODE
    if intent == "query_fact":
        name = next((f["value"] for f in facts if f["key"] == "name"), None)

        if name:
            response = f"Tu t'appelles {name}"
        else:
            response = "Je ne connais pas encore ton prénom."

        add_message("assistant", response)
        return response

    # 7. DEFAULT CHAT MODE
    prompt = (
        SYSTEM_PROMPT
        + "\n\nFACTS:\n"
        + (facts_text if facts_text else "- none")
        + "\n\nHISTORY:\n"
        + history_text
        + "\n\nUSER:\n"
        + user_input
    )

    response = call_llm(prompt)

    add_message("assistant", response)
    return response
