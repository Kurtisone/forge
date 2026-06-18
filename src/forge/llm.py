from forge.config import (
    FORGE_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
    LLM_MODEL
)

from forge.providers.llm_provider import (
    call_ollama,
    call_openrouter
)

def call_llm(prompt: str):

    if FORGE_PROVIDER == "local":
        return call_ollama(prompt, LLM_MODEL)

    if FORGE_PROVIDER == "openrouter":
        return call_openrouter(
            OPENROUTER_URL,
            OPENROUTER_API_KEY,
            LLM_MODEL,
            prompt
        )

    raise ValueError(f"Unknown provider: {FORGE_PROVIDER}")
