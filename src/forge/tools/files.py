"""
Placeholder for the files tool (see README roadmap).

Intentionally empty: it exposes no run() function, so the registry
skips it. When this gets implemented:

1. Add a run(content: str) -> str function here.
2. It will NOT become reachable just by existing -- it also has to
   be added to ENABLED_TOOLS (config.py), which defaults to
   "chat,code" only. This is intentional: a files tool can execute
   side effects, so enabling it is an explicit opt-in, not a side
   effect of writing code.
3. Add sandboxing (timeouts, restricted scope) before enabling it for
   real use -- the registry guard above is necessary, not sufficient.
"""
