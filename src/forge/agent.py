from forge.llm import call_llm
from forge.memory import add_message, get_history


SYSTEM_PROMPT = """
Tu es Forge, un assistant local pour développeur.
Réponds de manière claire et structurée.
"""


def run_agent(user_input: str):
    # 1. sauvegarde user input
    add_message("user", user_input)

    # 2. récupère historique
    history = get_history()

    # 3. construit le contexte
    history_text = ""

    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = (
        SYSTEM_PROMPT
        + "\n\nConversation history:\n"
        + history_text
        + "\nUser request:\n"
        + user_input
    )

    # 4. appel LLM
    response = call_llm(prompt)

    # 5. sauvegarde réponse
    add_message("assistant", response)

    return response
