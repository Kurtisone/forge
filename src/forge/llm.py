from forge.config import (
    FORGE_PROVIDER,
    OLLAMA_URL,
    OPENROUTER_URL,
    OPENROUTER_API_KEY,
    LLAMA_CPP_URL,
    LLM_MODEL
)

from forge.providers.llm_provider import (
    call_ollama,
    call_openrouter,
    call_llama_cpp
)

def call_llm(prompt: str):

    provider = FORGE_PROVIDER

    if provider == "ollama":
        return call_ollama(OLLAMA_URL, LLM_MODEL, prompt)

    if provider == "llama_cpp":
        return call_llama_cpp(LLAMA_CPP_URL, LLM_MODEL, prompt)

    if provider == "openrouter":
        return call_openrouter(
            OPENROUTER_URL,
            OPENROUTER_API_KEY,
            LLM_MODEL,
            prompt
        )

    raise ValueError(f"Unknown provider: {provider}")
