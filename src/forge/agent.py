from forge.llm import call_llm

SYSTEM_PROMPT = """
Tu es Forge, un assistant local pour développeur.
Réponds de manière claire et structurée.
"""

def run_agent(user_input: str):
    prompt = SYSTEM_PROMPT + "\n\nUser request:\n" + user_input
    return call_llm(prompt)
