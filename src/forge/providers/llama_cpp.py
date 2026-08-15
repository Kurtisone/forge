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
from forge.providers import error_body
from forge.types import Completion, Usage


def get_loaded_model(url: str) -> str | None:
    """
    Ask llama-server what it actually has loaded, via its own /props
    endpoint -- LLM_MODEL is never sent in the /completion payload
    (see call() above), so it's just a label Forge is trusting the
    person to keep in sync by hand. This queries the source of truth
    instead, for /health to display. Best-effort: returns None on any
    failure (server down, unexpected response shape, older llama.cpp
    build without /props) so callers can fall back to the configured
    LLM_MODEL rather than breaking /health over a label.
    """
    try:
        r = requests.get(f"{url}/props", timeout=2)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None

    # Field name/shape has drifted across llama.cpp server versions --
    # check the candidates in order rather than pinning to one.
    path = (
        data.get("model_path")
        or data.get("default_generation_settings", {}).get("model")
        or data.get("model")
    )
    if not path or not isinstance(path, str):
        return None

    return path.rsplit("/", 1)[-1]


def call(url: str, model: str, prompt: str) -> Completion:
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
    except requests.RequestException as e:
        # Never reached the server: connection refused, DNS, timeout.
        # There is no body to report here, only the transport error.
        raise ProviderError(f"llama_cpp request failed: {e}") from e

    try:
        r.raise_for_status()
    except requests.RequestException as e:
        # Reached the server and was rejected -- the body carries the
        # reason (see providers.error_body).
        raise ProviderError(f"llama_cpp request failed: {e}{error_body(r)}") from e

    data = r.json()

    # tokens_cached ("n_past" in llama.cpp's own terms) does not
    # reliably mean "tokens reused from this prompt" across server
    # versions/forks -- real-world testing here showed it exceeding
    # tokens_evaluated, and it's documented differently across
    # llama.cpp mirrors. Token counts alone aren't a trustworthy cache
    # signal, so this logs a timing-based one instead: prompt
    # processing time per prompt token. A shared prefix that's
    # actually being reused shows up as a sharp drop in ms/token on
    # the second+ call in a conversation versus the first (a "flat"
    # ms/token across calls means the cache isn't helping, regardless
    # of what tokens_cached claims).
    timings = data.get("timings", {})
    prompt_n = data.get("tokens_evaluated", timings.get("prompt_n"))
    prompt_ms = timings.get("prompt_ms")
    tokens_cached = data.get("tokens_cached")  # informational only
    if prompt_n:
        log.event(
            "llama_cpp.cache",
            prompt_n=prompt_n,
            prompt_ms=prompt_ms,
            ms_per_token=round(prompt_ms / prompt_n, 2) if prompt_ms else None,
            tokens_cached=tokens_cached,
        )

    content = data.get("content") or data.get("completion")
    if not content:
        raise ProviderError(f"llama_cpp returned no content: {data}")

    # prompt_n is already resolved above (tokens_evaluated, falling back
    # to timings.prompt_n) for the cache log -- reuse it rather than
    # re-deriving it with a different precedence.
    return Completion(
        text=content,
        usage=Usage(
            prompt_tokens=prompt_n,
            completion_tokens=data.get("tokens_predicted", timings.get("predicted_n")),
            cached_tokens=tokens_cached,
        ),
    )
