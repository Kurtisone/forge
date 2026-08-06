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

    results = data.get("results", [])[:SEARXNG_MAX_RESULTS]
    log.event("web_search.done", query=query[:120], results=len(results))
    return results


def _format_results(query: str, results: list[dict]) -> str:
    if not results:
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
