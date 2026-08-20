"""
Sandboxed web-search tool, backed by a self-hosted SearXNG instance.

Distinct from web_fetch: this queries a search index for ranked
results (title, URL, snippet) given a query string -- it does NOT
fetch a page whose URL is already known. Use web_search when the URL
isn't known ("latest news about X"); use web_fetch once a specific,
relevant URL has been identified (from a search result, or given
directly by the user). For a fully synthesized answer instead of a
raw results list, see graphs/research.py (dispatchable as the
"research" tool) -- it runs search -> fetch top N -> synthesize as
one deterministic sequence, without depending on the router to chain
multiple steps correctly (see its module docstring for why that
matters).

Requires a running SearXNG instance -- not a cloud search API, kept
consistent with Forge's self-hosting posture (same reasoning as
running its own llama.cpp/embedding servers rather than calling a
cloud LLM API). SearXNG's own settings.yml must have "json" added to
search.formats: this is disabled by default upstream specifically to
discourage scraping of PUBLIC SearXNG instances, but is the intended,
documented way to use a private, self-hosted one that only Forge
talks to.

To activate: ENABLED_TOOLS=chat,code,web_search in .env.local
To configure: SEARXNG_URL=http://<host>:8888 (default assumes a local
  instance on the same host/network as Forge)

Interface: run(content: str) -> str
  content is the search query, e.g. "actualités bourse aujourd'hui"
"""

import threading

import requests

from forge.config import SEARXNG_MAX_RESULTS, SEARXNG_TIMEOUT, SEARXNG_URL
from forge.kernel.capability import Requirements
from forge.logger import log

# Reaches the Internet through the self-hosted SearXNG instance.
REQUIREMENTS = Requirements(
    network=True,
    llm=False,
    mutates_workspace=False,
    spawns_process=False,
)


_MAX_SNIPPET_CHARS = 300

# Names of the engines SearXNG itself reported as down on the last
# search, as a side channel rather than a second return value.
#
# SearXNG answers 200 with results:[] when every engine behind it
# failed, which is byte-identical to a query that genuinely matched
# nothing. Forge was throwing away the one field that tells them
# apart, so a dead backend read as "aucun résultat" -- and on
# 2026-08-19 that cost a diagnosis: all five engines were timing out
# on a DNS fault and nothing said so.
#
# A side channel, because search() returning a tuple would change the
# contract of the one function graphs/research.py depends on, for
# information most callers have no use for. Thread-local for the same
# reason turn.py is: api.py serves turns from a two-worker pool, and a
# plain module global would let one run report the other run's dead
# engines.
_local = threading.local()


def last_unresponsive() -> list[str]:
    """Engines that failed during the most recent search() ON THIS
    THREAD, or [] outside a search. Only meaningful immediately after
    a search() call -- it is not a health history."""
    return list(getattr(_local, "unresponsive", ()))


def _normalize_unresponsive(raw: object) -> list[str]:
    """
    SearXNG's unresponsive_engines is not one shape. Depending on the
    version it holds ["google", ...], [["google", "timeout"], ...] or
    [{"engine": "google", ...}, ...]. All three are read here, and
    anything else is reduced to str() rather than dropped: an engine
    name Forge failed to parse is still evidence the backend fell
    over, which is the whole point of reading this field.

    Never raises. This runs on the success path of every search, and a
    diagnostic that can break the thing it diagnoses is worse than no
    diagnostic.
    """
    if not isinstance(raw, list):
        return []

    names: list[str] = []
    for entry in raw:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = str(entry.get("engine") or entry.get("name") or entry)
        elif isinstance(entry, (list, tuple)) and entry:
            name = str(entry[0])
        else:
            name = str(entry)
        name = name.strip()
        if name:
            names.append(name)
    return names


class SearchError(Exception):
    """Raised by search() -- structured-result callers (e.g.
    graphs/research.py) want to handle failure themselves rather than
    receive a "[error] ..." string mixed in with real results."""


def search(query: str) -> list[dict]:
    """
    Query SearXNG and return raw results as a list of
    {"title": ..., "url": ..., "content": ...} dicts, capped at
    SEARXNG_MAX_RESULTS. Raises SearchError on any failure. This is
    the structured form used by graphs/research.py; run() below
    formats the same data as a display string for direct chat/router
    dispatch.
    """
    query = query.strip()
    # Cleared first, not on the way out: every path below can raise,
    # and a stale list from an earlier search on this thread would
    # blame engines that had nothing to do with this query.
    _local.unresponsive = []
    if not query:
        raise SearchError("empty search query")

    log.event("web_search.run", query=query[:120])

    try:
        resp = requests.get(
            f"{SEARXNG_URL.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            timeout=SEARXNG_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        raise SearchError(f"search timed out after {SEARXNG_TIMEOUT}s") from None
    except requests.exceptions.RequestException as e:
        raise SearchError(f"search request failed: {e}") from e

    if resp.status_code != 200:
        raise SearchError(f"SearXNG returned HTTP {resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        raise SearchError(
            'SearXNG did not return JSON -- add "json" to search.formats '
            "in its settings.yml (disabled by default upstream)"
        ) from None

    unresponsive = _normalize_unresponsive(data.get("unresponsive_engines"))
    _local.unresponsive = unresponsive

    results = data.get("results", [])[:SEARXNG_MAX_RESULTS]
    log.event(
        "web_search.done",
        query=query[:120],
        results=len(results),
        unresponsive=len(unresponsive),
    )
    if unresponsive:
        # Warning even when results came back: a partial outage still
        # narrows what was searched, and the ranking above is not the
        # ranking the query would have had.
        log.warning(
            "web_search: SearXNG reported %d unresponsive engine(s): %s",
            len(unresponsive),
            ", ".join(unresponsive),
        )
    return results


def _format_results(query: str, results: list[dict]) -> str:
    if not results:
        down = last_unresponsive()
        if down:
            # Deliberately not "[no results]". The two are the same
            # HTTP response and mean opposite things: one says the web
            # has no answer, the other says Forge did not get to look.
            # Told the first, the model answers from its own weights
            # and sounds just as confident.
            return (
                f"[error] search backend failure for query {query!r}: "
                f"no engine answered ({len(down)} down: {', '.join(down)}). "
                "This is not an empty result -- the search did not run."
            )
        return f"[no results] for query: {query!r}"

    lines = [f"Search results for {query!r}:"]
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        snippet = (r.get("content") or "").strip()
        if len(snippet) > _MAX_SNIPPET_CHARS:
            snippet = snippet[:_MAX_SNIPPET_CHARS].rstrip() + "…"
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n".join(lines)


def run(content: str) -> str:
    query = content.strip()
    try:
        results = search(query)
    except SearchError as e:
        return f"[error] {e}"
    return _format_results(query, results)
