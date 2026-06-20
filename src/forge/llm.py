from forge.config import FORGE_PROVIDER, OLLAMA_URL, LLAMA_CPP_URL, LLM_MODEL
from forge.providers.llm_provider import call_ollama, call_llama_cpp

def call_llm(prompt: str):
    if FORGE_PROVIDER == "ollama":
        return call_ollama(OLLAMA_URL, LLM_MODEL, prompt)

    if FORGE_PROVIDER == "llama_cpp":
        return call_llama_cpp(LLAMA_CPP_URL, LLM_MODEL, prompt)

    raise ValueError(f"Unknown provider: {FORGE_PROVIDER}")
