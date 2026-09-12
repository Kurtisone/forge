import requests

from forge import gbnf
from forge.config import (
    LLAMA_CPP_APPLY_TEMPLATE,
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

# The sentinel handed to /apply-template so the constant framing can be
# split off it. Deliberately ugly and deliberately free of characters a
# Jinja template escapes or a tokenizer splits on in an interesting way.
_SENTINEL = "FORGEPROMPTSENTINEL"

# (prefix, suffix) for the model currently loaded, or None for "not
# asked yet". Cached for the life of the process because the answer is
# a property of the loaded model and re-asking it every call would put
# an HTTP round-trip in front of every routing decision. The cost of
# that cache is that swapping the served model under a running Forge
# leaves the old framing in place until it restarts -- the same
# trade-off, and the same remedy, as any other process-lifetime cache.
_template_framing: tuple[str, str] | None = None


def _framing(url: str) -> tuple[str, str]:
    """
    What the loaded model wants around a user turn, asked rather than
    assumed.

    llama-server's /apply-template renders the model's OWN chat
    template, so nothing here knows or cares whether that is ChatML,
    Llama-3, or something that does not exist yet. Send a sentinel as
    the only message, and whatever comes back on either side of it is
    the framing.

    Best-effort, like get_loaded_model: ("", "") on any failure -- an
    older llama.cpp with no such endpoint, a server that is down, a
    template that swallowed the sentinel -- so a provider that cannot
    answer the question degrades to the raw prompt Forge sent before
    this existed, rather than failing the turn over a wrapper.
    """
    global _template_framing
    if _template_framing is not None:
        return _template_framing

    framing = ("", "")
    try:
        r = requests.post(
            f"{url}/apply-template",
            json={"messages": [{"role": "user", "content": _SENTINEL}]},
            timeout=5,
        )
        r.raise_for_status()
        rendered = r.json().get("prompt")
    except (requests.RequestException, ValueError):
        rendered = None

    # Exactly once, or the split is meaningless: a template that drops
    # the sentinel gives ("", whole thing) and one that repeats it in a
    # system preamble would put half the framing in the wrong half.
    if isinstance(rendered, str) and rendered.count(_SENTINEL) == 1:
        prefix, suffix = rendered.split(_SENTINEL)
        framing = (prefix, suffix)
        log.info(
            "chat template: %d chars before the prompt, %d after",
            len(prefix),
            len(suffix),
        )
    else:
        log.warning("chat template unavailable at %s, sending the raw prompt", url)

    _template_framing = framing
    return framing


def reset_template_cache() -> None:
    """Forget the framing, so the next call asks again. For tests."""
    global _template_framing
    _template_framing = None


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


def get_context_size(url: str) -> int | None:
    """
    The context window llama-server was actually started with, from the
    same /props endpoint as get_loaded_model.

    Asking the server rather than reading a config value is the whole
    point: -c is passed on the llama-server command line, in a compose
    file Forge does not read, so any value Forge held would be a copy
    that drifts silently. A gauge showing a denominator nobody
    maintains is worse than no gauge.

    Best-effort in the same way and for the same reasons: None on any
    failure, so a caller falls back rather than breaking over it.
    """
    try:
        r = requests.get(f"{url}/props", timeout=2)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        return None

    # Same version drift as the model path above -- n_ctx has lived at
    # the top level and inside default_generation_settings depending on
    # the build.
    n_ctx = data.get("n_ctx") or data.get("default_generation_settings", {}).get(
        "n_ctx"
    )
    if not isinstance(n_ctx, int) or n_ctx <= 0:
        return None

    return n_ctx


def _grammar_for(grammar: str | None) -> str | None:
    """
    Which grammar text to send, or None to send none.

    LLAMA_CPP_USE_GRAMMAR stays absolute: when it's off, nothing is
    constrained, whatever the caller asked for -- it exists to take
    grammar sampling out of the picture while debugging, and a knob
    with an exception isn't one.

    An invalid grammar is dropped rather than sent. llama-server
    answers 400 to every completion it can't parse, so sending one is
    a guaranteed total outage; running unconstrained falls back to
    router/parser.py's existing chain, which is degraded but alive.
    The log line carries the offending rule name, which the server's
    own 400 body never does.
    """
    if not LLAMA_CPP_USE_GRAMMAR:
        return None

    if grammar is None:
        # The router's grammar for a caller that named none, including
        # the four graph syntheses whose prompts ask for plain text.
        # That looks like a bug and is not one. MEASURED ON THE DECK,
        # 2026-08-21, by giving each of them a prose grammar instead:
        #
        #   recall     '<answer>' and nothing else, 8 chars
        #   review     'NO_THINK:', 5 tokens, then stop
        #   research   1536 tokens, hit n_predict, output was the
        #              PROMPT's own instructions read back
        #   sysadmin   '<analysis>' block, ~2x the tokens
        #
        # against a router arm that answered correctly every time. The
        # grammar was never only stopping JSON. It was suppressing the
        # scaffolding this model reaches for -- <answer>, <think>,
        # 'NO_THINK:', prompt echo -- and it was the only hard
        # TERMINATOR in the loop, since "{...}" cannot run past its
        # closing brace and free prose can run to n_predict.
        #
        # bench/prose_grammar_ab.py reproduces the whole comparison.
        # The graph prompts have never been written for unconstrained
        # decoding, and swapping the grammar without rewriting them
        # trades a cosmetic log warning for empty answers.
        #
        # So: try_unwrap_router_json() in the graphs is not a
        # workaround to be removed. It is the seam that makes this
        # fallback safe, and tests/test_graph_grammar.py pins it.
        from forge.router.grammar import build_router_grammar

        grammar = build_router_grammar()

    try:
        gbnf.validate(grammar)
    except gbnf.GrammarError as e:
        log.error("grammar rejected before sending, running unconstrained: %s", e)
        return None

    return grammar


def call(url: str, model: str, prompt: str, grammar: str | None = None) -> Completion:
    if LLAMA_CPP_APPLY_TEMPLATE:
        prefix, suffix = _framing(url)
        prompt = f"{prefix}{prompt}{suffix}"

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

    # Grammar-constrained decoding: the model can only emit tokens
    # matching a JSON schema at the sampling level -- it cannot
    # hallucinate a new "User:" turn, leak prompt text, or emit
    # anything but valid JSON in the first place. The stop sequences
    # above stay as defense-in-depth (a model can still choose not to
    # stop generating right after a complete, valid object) rather
    # than the primary safeguard.
    #
    # `grammar` defaults to the router's (router/grammar.py), which is
    # what every call used until v3.13. A caller passing its own is
    # how a graph gets a shape that ISN'T a routing decision -- before
    # this, the router grammar applied to every call including graph
    # syntheses, which is why the graphs all carry
    # try_unwrap_router_json().
    grammar_text = _grammar_for(grammar)
    if grammar_text is not None:
        payload["grammar"] = grammar_text

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
