"""
Dispatchable wrapper around forge.graphs.recall.

Forge's UI is a single conversational page, zero tabs -- recall must
be reachable by just asking for it in chat, the same reasoning as
tools/research.py. Same content contract as research: one plain-text
field (the question), not JSON -- unlike tools/memory.py's "recall"
action, which stays JSON because it also carries "remember".

Interface: run(content: str) -> str
  content is the question, e.g. "Tu peux me lister mon matériel ?"

To activate: ENABLED_TOOLS=chat,code,memory,recall in .env.local
Requires memory's own dependency: a reachable embedding server
(EMBEDDING_URL) -- recall calls forge.tools.memory.search() directly,
it does not require "memory" itself to also be listed in
ENABLED_TOOLS.
"""

from forge.graphs.recall import run as recall_run


def run(content: str) -> str:
    query = content.strip()
    if not query:
        return "[error] empty recall query"
    return recall_run(query)
