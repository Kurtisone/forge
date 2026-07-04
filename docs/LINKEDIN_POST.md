# LinkedIn post drafts — Forge

Two lengths, same story. Pick one, tweak the voice to sound like you, post
whichever fits how much people usually read from you.

Before posting: export the architecture diagram from the README as an image
(a screenshot of the rendered Mermaid block works fine, or `mmdc -i diagram.mmd
-o diagram.png` if you have mermaid-cli) — LinkedIn posts with an image get
read a lot more than a wall of text.

---

## Short version

I've spent the last few weeks building **Forge**, a small LLM agent runtime,
from scratch — and running entirely on a Steam Deck.

The constraint shaped every decision: no GPU cluster, no API budget to burn on
every experiment, just llama.cpp doing ~7 tokens/sec locally. So instead of a
big autonomous reasoning loop, Forge routes each request through a small LLM
call that outputs strict JSON (`{"tool": "...", "content": "..."}`), and a
deterministic dispatcher executes exactly that — chat, code, sandboxed file
ops, sandboxed shell, read-only git. No hidden reasoning, no surprises, and it
still works reliably on hardware most agent frameworks would consider a joke.

Commits later: a resilient router (JSON → XML → markdown-fence → plain-text
fallback chain), a graph engine for multi-step workflows, a FastAPI + web UI
layer, optional multi-step execution the model can opt into, CI, and optional
API auth for anyone running it beyond localhost.

Code: [github.com/Kurtisone/forge](https://github.com/Kurtisone/forge)

#LLM #AgentFramework #SideProject #Python #LocalLLM

---

## Longer version

**Building an LLM agent framework on a Steam Deck taught me more about
architecture than any cloud GPU would have.**

A few weeks ago I started Forge: a lightweight agent runtime built around one
idea — route, don't reason. Instead of asking a model to autonomously decide
and chain actions in a big opaque loop, Forge asks it exactly one structured
question per step: *"which tool, with what content?"* The model answers with
JSON. A separate, fully deterministic dispatcher does the actual work.

Why so strict? Because I was running this on a Steam Deck with llama.cpp —
~7 tokens/sec, no cloud fallback, no room for a reasoning loop that
occasionally goes sideways and burns the whole budget re-litigating itself.
Constraints forced clarity:

- **The router never executes anything.** It only decides.
- **Tools are explicitly allowlisted** (`ENABLED_TOOLS`) — a module having a
  `run()` function doesn't make it reachable from model output. That has to
  be opted into on purpose.
- **The extraction chain is resilient by design**: JSON first, then an XML
  tool-call fallback (for models fine-tuned differently), then markdown
  fences, then plain text — because small local models don't always follow
  format instructions perfectly, and failing loudly on that would make the
  whole thing unusable.
- **Multi-step is opt-in, not default.** The orchestrator only goes beyond one
  step if the router explicitly says `"done": false` — everything before that
  stays single-shot and predictable.

What it does today: a resilient router, a tool dispatcher (chat, code,
sandboxed file ops, sandboxed shell commands, read-only git), a graph engine
for expressing multi-step workflows as typed nodes and conditional edges, a
FastAPI server with a small web UI, execution tracing to JSONL, rolling
memory, CI (ruff + pytest on every push), and optional bearer-token auth for
anyone running the API beyond localhost.

None of this needed a GPU cluster. It needed the discipline of building for
the weakest hardware I own, first.

Repo, README with the full architecture diagram, and the roadmap:
[github.com/Kurtisone/forge](https://github.com/Kurtisone/forge)

#LLM #AgentFramework #LocalLLM #Python #FastAPI #SoftwareArchitecture #SideProject
