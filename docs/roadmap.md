# Version history and roadmap

The product roadmap. It is deliberately separate from
[ARCHITECTURE.md](../ARCHITECTURE.md): the architectural maturity axis
(Kernel, capabilities, scheduler) advances independently of releases and
conflating the two produced a version number that meant two things.

## Roadmap

| Version | Status | Focus |
|---|---|---|
| **v2.2** | done | Clean Runtime: typed errors, centralized logger, provider split, loop guard |
| **v2.3** | done | Robustness: parser cascade, memory hardening, REPL paste detection, ENABLED_TOOLS allowlist |
| **v2.4** | done | Structured execution trace: `AgentState`, `TraceStep`, JSONL trace file, `!trace` |
| **v3.0** | done | Graph execution engine: `Node/Edge/Graph`, conditional edges, `AgentState.context` |
| **v3.1** | done | HTTP API + web UI, review graph, `forge review` CLI, sandboxed files tool |
| **v3.2** | done | Shell tool, git tool, `POST /run`, Tools tab in UI |
| **v3.3** | done | Hardening: real multi-step orchestrator, CI (ruff + pytest), optional API bearer-token auth |
| **v3.4** | done | Portfolio: architecture diagram, `.env.example`, LinkedIn writeup |
| **v3.5** | done | Test coverage (llm/cli/trace: 26-39% → 98-100%), router reachable to files/shell/git, API rate limiting |
| **v3.6** | done | Response quality: GBNF grammar-constrained decoding for llama.cpp |
| **v3.7** | done | Vector memory / RAG: SQLite-vec, `/remember` + `/search`, `!remember`/`!recall` REPL commands, a router-dispatchable `memory` tool, Qwen3-Embedding-0.6B |
| **v3.8** | done | Prompt-cache reliability: pinned llama-server slot, `MEMORY_MAX_HISTORY` raised to stop a sliding window from fighting KV-cache reuse — root-caused a remaining cache-reuse gap to the served model's own hybrid architecture, not Forge |
| **v3.9** | done | Context compaction + drawer: `rag_pointer`/`llm_summary` strategies, pin/unpin, `/history` `/drawer` `/compact` endpoints, `!compact` REPL command, files write-diff |
| **v3.10** | done | Hardening + new tools: dedicated `test` tool, `web_fetch` (SSRF-guarded), `web_search` + `research` (self-hosted SearXNG), review graph gains an optional test-run step and chat-dispatch; router disambiguation fixes (files vs review, tool descriptions/examples for every new tool) found through real usage |
| **v3.11** | done | Sysadmin: `discover → collect → synthesize` graph diagnosing real service/system problems from logs, read-only always (no restart/stop path exists in the code); UI gains expandable per-step detail (`forge.subtrace`) for every graph-based tool; read-only host access via three independent proxies (`xdg-dbus-proxy` for systemd, a hand-rolled GET-only proxy for podman.sock, a plain bind mount for the journal) rather than raw socket access — real production debugging found and fixed a prompt-injection-shaped example-leak, a context-overflow crash, `systemctl`'s undocumented refusal to honor `DBUS_SYSTEM_BUS_ADDRESS` (switched discovery to `busctl`), and a rootless-podman supplementary-groups gap blocking `journalctl -u` on root-owned services (`--group-add keep-groups`) |
| **v3.12** | done | Instrumentation + router prompt latency, five batches: per-call token accounting from llama-server's own timings; the router prompt made a **pure append** over the previous turn, taking warm prompt processing from 3.12 to 0.81 ms/token (~3.9x, ~8.4s to ~2.2s per routing call) with zero regressions across a 29-fixture A/B (`bench/router_ab.py`); a permanent context gauge in the header with `GET /context`; compaction triggered on a token budget rather than a message count; and file-path grounding in the orchestrator, refusing a write or a review on a path the model invented |
| **v3.13** | done | Delegation to Claude Code: a spec drafted under its own GBNF grammar, persisted jobs with a checked lifecycle (`draft → awaiting_user → ready → running → done`), the model asking follow-up questions rather than inventing missing detail, cancellation, and a runner thread that hands off without blocking the conversation. Entry is a `delegate` tool, so it stays reachable from the chat thread like everything else |
| **fix/dettes-v3.12** | done | Six fixes found by running the previous two versions in anger: `test_path` was invisible to the path guard (and so was a raw `pytest <path>` command string), a pasted paragraph in `file_path` is text rather than an unfounded path, `load_memory()` died on valid JSON of the wrong shape, `recall` answered in the wrong language (new `forge/lang.py`, closed-vocabulary fr/en detection that stays silent when unsure), and the `test` tool had neither a description nor an example in the router prompt |
| **Kernel L2** | this branch | Capability layer and a deterministic Policy Engine — see [ARCHITECTURE.md](../ARCHITECTURE.md) and [The Kernel layer](architecture.md#the-kernel-layer). Sits on the architectural maturity axis, not this product roadmap |

---

[← Documentation index](README.md) · [← Project README](../README.md)
