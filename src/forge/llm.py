"""
LLM dispatch layer.

This is the ONLY module the orchestrator talks to for inference. It
knows nothing about tools, routing, or logging policy -- it just
turns a prompt into text, or raises ProviderError. That is the
"LLM" leg of the LLM / tools / logs separation.

Being the single dispatch point also makes it the single place where
what a run costs can be observed. Providers hand back a Completion
(text + Usage); call_llm keeps returning a plain str so nothing
upstream changes, and records the usage on the way through.
"""

import time

from forge import metrics
from forge.config import (
    FORGE_PROVIDER,
    LLAMA_CPP_URL,
    LLM_MODEL,
    OLLAMA_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_URL,
)
from forge.errors import ProviderError
from forge.logger import log
from forge.providers import llama_cpp, ollama, openrouter


def call_llm(prompt: str) -> str:
    started = time.monotonic()
    log.event("llm.call", provider=FORGE_PROVIDER, model=LLM_MODEL)

    try:
        if FORGE_PROVIDER == "ollama":
            result = ollama.call(OLLAMA_URL, LLM_MODEL, prompt)
        elif FORGE_PROVIDER == "llama_cpp":
            result = llama_cpp.call(LLAMA_CPP_URL, LLM_MODEL, prompt)
        elif FORGE_PROVIDER == "openrouter":
            result = openrouter.call(
                OPENROUTER_URL, OPENROUTER_API_KEY, LLM_MODEL, prompt
            )
        else:
            raise ProviderError(f"Unknown provider: {FORGE_PROVIDER!r}")
    except ProviderError:
        raise
    except Exception as e:
        raise ProviderError(f"unexpected provider failure: {e}") from e

    elapsed_ms = int((time.monotonic() - started) * 1000)
    metrics.record(result.usage, elapsed_ms)
    log.event(
        "llm.response",
        elapsed_ms=elapsed_ms,
        length=len(result.text),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
    )
    return result.text
