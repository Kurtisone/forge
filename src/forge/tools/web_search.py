"""
Sandboxed web-search tool, backed by a self-hosted SearXNG instance.

Distinct from web_fetch: this queries a search index for ranked
results (title, URL, snippet) given a query string -- it does NOT
fetch a page whose URL is already known. Use web_search when the URL
isn't known ("latest news about X"); use web_fetch once a specific,
relevant URL has been identified (from a search result, or given
directly by the user).

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
from forge.logger import log

_MAX_SNIPPET_CHARS = 300


def run(content: str) -> str:
    query = content.strip()
    if not query:
        return "[error] empty search query"

    log.event("web_search.run", query=query[:120])

    try:
        resp = requests.get(
            f"{SEARXNG_URL.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            timeout=SEARXNG_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return f"[error] search timed out after {SEARXNG_TIMEOUT}s"
    except requests.exceptions.RequestException as e:
        return f"[error] search request failed: {e}"

    if resp.status_code != 200:
        return f"[error] SearXNG returned HTTP {resp.status_code}"

    try:
        data = resp.json()
    except ValueError:
        return (
            '[error] SearXNG did not return JSON -- add "json" to '
            "search.formats in its settings.yml (disabled by default "
            "upstream)"
        )

    results = data.get("results", [])[:SEARXNG_MAX_RESULTS]
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

    log.event("web_search.done", query=query[:120], results=len(results))
    return "\n".join(lines)
