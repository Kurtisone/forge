"""
Provider backends for forge.llm.

Only forge.llm imports these -- nothing else in the runtime talks to a
provider directly, which is what keeps the "swap the LLM without
touching the rest" property real.
"""

_MAX_BODY = 500


def error_body(response, limit: int = _MAX_BODY) -> str:
    """
    Return the backend's own error body, formatted for appending to a
    ProviderError message (empty string when there is nothing useful).

    requests' HTTPError stringifies to the status line and the URL and
    nothing else -- the body is dropped. For llama-server that body is
    routinely the only place the actual cause appears: a malformed GBNF
    grammar comes back as a plain 400 whose body says exactly which
    rule failed to parse. Without this, such a failure surfaces as
    "400 Client Error: Bad Request", every completion in the run fails,
    and the run finishes in milliseconds looking deceptively like a
    fast one. Reading llama-server's own log by hand was the only way
    to find out why.

    Best-effort by construction: a response object with no readable
    .text (a mock, a streamed body already consumed) yields "" rather
    than turning an error report into a second error.
    """
    try:
        body = (response.text or "").strip()
    except Exception:  # noqa: BLE001 - never let error reporting raise
        return ""
    if not body:
        return ""
    if len(body) > limit:
        body = body[:limit] + f"... ({len(body)} bytes total)"
    return f" -- backend said: {body}"
