# Forge documentation

The project [README](../README.md) is the door: what Forge is, how to
start it, where things are. Everything past that lives here, one page
per subject, because a single 900-line file is a file nobody re-reads.

| Page | Read it when |
|---|---|
| [Architecture](architecture.md) | You want to know how a turn flows, or why a boundary is where it is |
| [Usage](usage.md) | You are running Forge -- REPL, web UI, CLI, container |
| [Configuration](configuration.md) | You are changing behaviour without changing code |
| [Tools](tools.md) | You want to know what a tool does, refuses, and costs |
| [Memory, RAG and traces](memory.md) | You are chasing something across the three stores |
| [HTTP API](api.md) | You are calling Forge from something other than the UI |
| [Development](development.md) | You are running the tests, CI, or a measurement harness |
| [Version history and roadmap](roadmap.md) | You want to know what landed when, and why |

Three documents stay at the root because they are not "docs about the
code" -- they are commitments:

- [ARCHITECTURE.md](../ARCHITECTURE.md) — the long-term direction
  (micro-kernel, capabilities, policy). Moves at a different speed
  than the code, deliberately.
- [SECURITY.md](../SECURITY.md) — the threat model, what is enforced
  in code rather than asked of the model, and the limits that are
  known and accepted.
- [.env.example](../.env.example) — the exhaustive configuration
  reference, with the reasoning behind each default.

And [deploy/README.md](../deploy/README.md) covers the read-only host
access design: the D-Bus proxy, the podman proxy, the journal mount.
