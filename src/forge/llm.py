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
from forge.tokens import estimate_tokens


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
    _check_estimate(prompt, result.usage.prompt_tokens)
    return result.text


# Estimation drift beyond this is worth a line in the log. Below it the
# estimate is doing its job and saying so every call would be noise.
_ESTIMATE_TOLERANCE_PCT = 15


def _check_estimate(prompt: str, actual: int | None) -> None:
    """
    Compare tokens.estimate_tokens to what the backend actually
    counted. This is the only place in Forge that sees both numbers for
    the same text, which makes it the only place the local estimator
    can be kept honest -- see the CALIBRATION block in tokens.py.

    Observation only: nothing downstream reads this, and a failure here
    must never cost a run. The estimator is biased high on purpose, so
    a positive error is expected and unremarkable; a NEGATIVE one means
    it is understating, which is the direction that overflows a context
    window.
    """
    # Falsy covers both None (backend reported nothing) and 0 (nothing
    # to divide by). Neither is a drift signal.
    if not actual:
        return

    estimated = estimate_tokens(prompt)
    error_pct = round((estimated - actual) / actual * 100, 1)
    if abs(error_pct) >= _ESTIMATE_TOLERANCE_PCT:
        log.event(
            "tokens.estimate_drift",
            chars=len(prompt),
            estimated=estimated,
            actual=actual,
            error_pct=error_pct,
        )
