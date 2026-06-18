import re
from forge.llm import call_llm
from forge.memory import add_message, get_history, add_fact, get_facts


SYSTEM_PROMPT = """
Tu es Forge, un assistant local pour développeur.
Réponds de manière claire et structurée.
"""


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

    # pattern 1 : nom
    match = re.search(r"je m'appelle (.+)", text)
    if match:
        name = match.group(1).strip()
        add_fact("user_name", name)
        return

    # pattern 2 : j'habite
    match = re.search(r"j'habite (.+)", text)
    if match:
        location = match.group(1).strip()
        add_fact("user_location", location)
        return

    # pattern 3 : j'aime
    match = re.search(r"j'aime (.+)", text)
    if match:
        like = match.group(1).strip()
        add_fact("user_like", like)
        return
