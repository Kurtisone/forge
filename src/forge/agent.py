import json, re
from forge.llm import call_llm
from forge.memory import add_message, get_history, update_fact, get_facts
from forge.config import SHOW_REASONING

SYSTEM_PROMPT = """
You are Forge, a coding assistant.

RULES:
- Return ONLY the answer requested by the user
- Do NOT add explanations unless asked
- Do NOT add tests unless asked
- Do NOT add markdown titles or sections
- Do NOT include system-like text

OUTPUT STYLE:
- If user asks code → return code only
- If user asks explanation → short explanation only

"""

# ----------------------------
#PARSE REASONING
# ----------------------------
def parse_llm_output(text: str):
    reasoning = ""
    answer = text

    match = re.search(r"\[REASONING\](.*?)\[/REASONING\]", text, re.DOTALL)

    if match:
        reasoning = match.group(1).strip()
        answer = re.sub(r"\[REASONING\](.*?)\[/REASONING\]", text, re.DOTALL).strip()

    return reasoning, answer

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
Return ONLY valid JSON:
{{"intent": "chat"}}

User:
{user_input}
"""

    try:
        raw = call_llm(prompt)
        data = json.loads(raw)
        if "intent" in data:
            return data
    except Exception:
        pass

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

    raw = call_llm(prompt)

    reasoning, response = parse_llm_output(raw)

    if SHOW_REASONING:
        print("\nREASONING:\n", reasoning)

    add_message("assistant", response)

    return response
