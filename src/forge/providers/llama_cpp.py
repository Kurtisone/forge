import requests

from forge.config import (
    LLAMA_CPP_CACHE_PROMPT,
    LLAMA_CPP_ID_SLOT,
    LLAMA_CPP_N_PREDICT,
    LLAMA_CPP_TIMEOUT,
    LLAMA_CPP_USE_GRAMMAR,
)
from forge.errors import ProviderError
from forge.logger import log


def call(url: str, model: str, prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": LLAMA_CPP_N_PREDICT,
        # Pin to a fixed slot and let llama-server reuse the KV cache
        # from the previous call's matching prefix instead of
        # recomputing it from scratch every turn (v3.8).
        "id_slot": LLAMA_CPP_ID_SLOT,
        "cache_prompt": LLAMA_CPP_CACHE_PROMPT,
        "stop": [
            # Prevent the model from hallucinating a new dialogue turn
            "\nUser:",
            "\nUser :",
            "User:",
            # Qwen HERETIC XML tool-call format: stop after the full
            # tool_call block (parser extracts <content> from it)
            "</tool_call>",
            # NOTE: "\n\n" intentionally absent — it would cut any
            # multi-line code response mid-generation before the
            # JSON or XML closing tag is reached.
        ],
    }

    if LLAMA_CPP_USE_GRAMMAR:
        # Grammar-constrained decoding: the model can only emit tokens
        # matching the router's exact JSON schema (see router/grammar.py),
        # at the sampling level -- it cannot hallucinate a new "User:"
        # turn, leak prompt text, or emit anything but valid JSON in the
        # first place. The stop sequences above stay as defense-in-depth
        # (a model can still choose not to stop generating right after a
        # complete, valid object) rather than the primary safeguard.
        from forge.router.grammar import build_router_grammar

        payload["grammar"] = build_router_grammar()

    try:
        r = requests.post(f"{url}/completion", json=payload, timeout=LLAMA_CPP_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        raise ProviderError(f"llama_cpp request failed: {e}") from e

    data = r.json()

    # timings.prompt_n is the number of prompt tokens actually
    # evaluated this call; cache_n (or timings.cache_n depending on
    # server version) is how many of the prompt tokens were served
    # from the reused KV cache instead. Logged as a ratio so cache
    # effectiveness can be tracked over time without parsing raw logs.
    timings = data.get("timings", {})
    prompt_n = data.get("prompt_n", timings.get("prompt_n"))
    cache_n = data.get("cache_n", timings.get("cache_n"))
    if prompt_n:
        log.event(
            "llama_cpp.cache",
            cache_n=cache_n,
            prompt_n=prompt_n,
            ratio=round((cache_n or 0) / prompt_n, 3),
        )

    content = data.get("content") or data.get("completion")
    if not content:
        raise ProviderError(f"llama_cpp returned no content: {data}")
    return content
