from forge.config import FORGE_PROVIDER, LLM_MODEL
from forge.providers.llm_provider import call_ollama, call_openrouter


PROVIDERS = {
    "local": call_ollama,
    "openrouter": call_openrouter
}


def call_llm(prompt: str):
    if FORGE_PROVIDER not in PROVIDERS:
        raise ValueError(f"Unknown provider: {FORGE_PROVIDER}")

    return PROVIDERS[FORGE_PROVIDER](prompt, LLM_MODEL)
