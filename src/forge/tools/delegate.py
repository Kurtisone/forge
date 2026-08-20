"""
Dispatchable wrapper around forge.graphs.delegate.

Same reasoning as tools/research.py and tools/sysadmin.py: Forge's UI
is one conversational page with zero tabs, so opening a delegation has
to be reachable by asking for it in chat. Content is plain text (the
request), not JSON -- there is only one field to pass, and the
structured part is what the graph produces, not what it receives.

To activate: ENABLED_TOOLS=chat,code,delegate in .env.local

Note that this tool is only the ENTRY point. Answering the questions
that follow, approving the spec and cancelling all happen in
delegation.py, above the router -- they are not tool calls, and adding
"delegate" to ENABLED_TOOLS does not change how they are handled.
"""

from forge.graphs.delegate import run as delegate_run
from forge.kernel.capability import Requirements

# Calls the LLM to draft the spec, and persists the job to JOBS_FILE
# -- which is data/, not WORKSPACE_DIR, so it is not a workspace write.
# The handoff to Claude Code happens outside this process.
REQUIREMENTS = Requirements(
    network=False,
    llm=True,
    mutates_workspace=False,
    spawns_process=False,
)


def run(content: str) -> str:
    request = content.strip()
    if not request:
        return "[error] empty delegation request"
    return delegate_run(request)
