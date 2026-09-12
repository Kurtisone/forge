# Forge documentation

The project [README](../README.md) is the door: what Forge is, how to
start it, where things are. Everything past that lives here, one page
per subject, because a single 900-line file is a file nobody re-reads.

That rule cost `memory.md` a split on 2026-09-12, when it reached 1282
lines and 55% of this directory. What moved out was not prose but a
*kind*: the measurement campaigns, which answer "why" and "what has
already been tried", now live in [campaigns.md](campaigns.md), and
`memory.md` keeps what is true today. Nothing was shortened — a dead
end is only worth writing down at the length that makes it
reproducible.

`roadmap.md` failed the same rule the same day, in a shape line counts
hide: 41 lines and 15 000 characters, because it was a table whose
cells had grown into essays — 4346 of them in v3.20's single cell. It
is an index again, with the detail in sections under it. The tell was
that all nine measurements it cited were already in `campaigns.md`: a
page drifts by taking on a neighbour's job, not by getting long.

| Page | Read it when |
|---|---|
| [Architecture](architecture.md) | You want to know how a turn flows, or why a boundary is where it is |
| [Usage](usage.md) | You are running Forge -- REPL, web UI, CLI, container |
| [Configuration](configuration.md) | You are changing behaviour without changing code |
| [Tools](tools.md) | You want to know what a tool does, refuses, and costs |
| [Memory, RAG and traces](memory.md) | You are chasing something across the three stores, or working with one |
| [The measurement record](campaigns.md) | You are about to change a retrieval default, or wondering why one ships off |
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
