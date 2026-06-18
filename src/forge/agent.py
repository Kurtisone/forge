import re
from forge.llm import call_llm
from forge.memory import add_message, get_history, add_fact, get_facts


SYSTEM_PROMPT = """
Tu es Forge, un assistant local pour développeur.
Réponds de manière claire et structurée.
"""

def normalize_value(value: str) -> str:
    value = value.strip()
    return value[0].upper() + value[1:] if value else value


def run_agent(user_input: str):

    # 1. extraction des facts AVANT tout
    extract_facts(user_input)

    # 2. sauvegarde user input
    add_message("user", user_input)

    # 3. récupération mémoire
    facts = get_facts()
    history = get_history()

    # 4. build facts text
    facts_text = ""
    for fact in facts:
        facts_text += f"- {fact['key']}: {fact['value']}\n"

    # 5. build history text
    history_text = ""
    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    # 6. prompt final
    prompt = (
        SYSTEM_PROMPT
        + "\n\nFACTS (persistent user knowledge):\n"
        + (facts_text if facts_text else "- none")
        + "\n\nConversation history:\n"
        + history_text
        + "\nUser request:\n"
        + user_input
    )

    # 7. appel LLM
    response = call_llm(prompt)

    # 8. sauvegarde réponse
    add_message("assistant", response)

    return response


def extract_facts(user_input: str):
    text = user_input.lower()

    # nom
    match = re.search(r"(je m'appelle|mon nom est|appelez-moi)\s+(.+)", text)
    if match:
        value = normalize_value(match.group(2))
        add_fact("user_name", value)

    # localisation
    match = re.search(r"(j'habite|je vis à|je suis à)\s+(.+)", text)
    if match:
        value = normalize_value(match.group(2))
        add_fact("user_location", value)

    # goûts
    match = re.search(r"j'aime\s+(.+)", text)
    if match:
        value = normalize_value(match.group(1))
        add_fact("user_like", value)
