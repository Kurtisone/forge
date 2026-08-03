"""
Dispatchable wrapper around forge.graphs.research.

Forge's UI is a single conversational page, zero tabs -- research
must be reachable by just asking for it in chat, the same reasoning
as tools/review.py. Unlike review's JSON content contract, research
only ever needs one field (the query), so content is plain text here,
not JSON.

Interface: run(content: str) -> str
  content is the search query, e.g. "actualités jeu vidéo"

To activate: ENABLED_TOOLS=chat,code,research in .env.local
Requires web_search's own dependency: a running SearXNG instance
(SEARXNG_URL) -- research calls forge.tools.web_search.search()
directly, it does not require "web_search" itself to also be listed
in ENABLED_TOOLS.
"""

from forge.graphs.research import run as research_run


def run(content: str) -> str:
    query = content.strip()
    if not query:
        return "[error] empty research query"
    return research_run(query)
