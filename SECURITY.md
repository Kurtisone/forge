# Security

Forge dispatches tools chosen by a language model, on a machine you
own, against data you don't always control. That combination is the
whole design, so this document says what it is assumed to protect
against and what it is not — before the interesting parts of the
roadmap (an Evolution Runtime that writes its own code) make those
assumptions much harder to change.

## Threat model

Forge is built for **one operator, one machine, private network
access**. Concretely:

- A single trusted user. There are no roles, no per-user separation,
  no audit trail attributable to a person.
- Reachable over WireGuard or localhost, not from the open internet.
  `API_TOKEN` is the only thing between a caller and `/chat`, and
  `/chat` dispatches tools.
- The **repository is public**; the deployment is not. Nothing in
  `main` may contain a secret, and configuration lives in
  `.env.local`, which is not tracked.
- The host is the operator's own (a Steam Deck today, a home server
  next). Forge is not a shared service, and hardening it into one
  would be a different project with a stricter version of this file.

**In scope** — things Forge is expected to resist:

- An unauthenticated caller who can reach the port.
- A hostile or compromised **web page** that Forge fetches, and a
  hostile **log line** that Forge reads. These reach the prompt that
  chooses the next tool.
- The model itself being wrong, degenerate, or steered. Every
  protection that matters is enforced in code, not asked of the model.

**Out of scope** — deliberately, not by oversight:

- A malicious operator. Anyone able to edit `.env.local` can enable
  `shell` with `python3` in its allowlist and has the machine.
- A compromised host, hypervisor, or LLM backend.
- The supply chain beyond pinning: `requirements.txt` and the base
  image are pinned to exact versions and a digest, which makes builds
  reproducible and unexpected upgrades visible. It does not audit
  what is in those versions.
- Denial of service. Rate limiting exists to stop accidental
  hammering, not a determined attacker.

## What is enforced in code

These are deterministic. None of them depend on the model behaving,
which matters because prompt wording has failed to steer this model
three separate recorded times on this project.

| Boundary | Where | What it guarantees |
| --- | --- | --- |
| Tool opt-in | `config.ENABLED_TOOLS`, `tools/registry.py` | A module with `run()` is not dispatchable until it is listed. |
| Auth | `config.API_TOKEN` | Forge refuses to start without a token unless `API_ALLOW_UNAUTHENTICATED=true` is written down explicitly. |
| Workspace confinement | `tools/files.py` `_safe_path`, `tools/review.py`, `tools/test.py` | Paths resolving outside `WORKSPACE_DIR` are rejected before any filesystem call. |
| SSRF | `tools/web_fetch.py` | Private, loopback and link-local resolved IPs are blocked. Not configurable, on purpose — Forge sits on a home network. |
| Escalation guard | `orchestrator.py` | Once a run has called `web_fetch`/`web_search`/`research`/`sysadmin`, no later step of that run may dispatch `shell`, `test`, or `files:write`. |
| Read-only host access | `deploy/podman_ro_proxy.py`, `deploy/forge-dbus-proxy.sh` | `sysadmin` can list units and read logs. Start/stop/exec are refused at the proxy, before Forge's own code is in a position to decide. |
| Loop guard | `orchestrator.py` | The same `(tool, content)` pair cannot be dispatched twice in one run. |
| Non-root container | `Containerfile`, `deploy/compose.example.yaml` | Runs as `forge:forge` (1000:1000). Requires `userns_mode: keep-id` at runtime; a test asserts the two halves stay coupled. |

## Known limits

Stated because a limit you know about is a decision, and one you don't
is a surprise.

**`files` + `test` is equivalent to `shell`.** Running pytest means
executing the Python code in the workspace — that is what a test
runner is. Write a file, then run it. `pytest` also auto-loads
`conftest.py` before collecting anything, so even an invocation naming
one innocuous file executes a `conftest.py` next to it. The allowlist
in `tools/test.py` restricts which binary starts, not what that binary
then executes; argument confinement bounds where that code may come
from, not that it runs. A warning is logged at startup when both tools
are enabled. Fixing this properly needs a disposable container per
run.

**The escalation guard is per-run, not per-session.** Content fetched
from a hostile page in one turn can be written to the workspace and
read back in a later turn, where the run starts untainted. Making the
taint persist would mean a fetch poisoning every subsequent turn until
something cleared it, with no obvious moment to clear it. Per-run is
the deliberate trade, not an oversight.

**`files:read` does not taint a run.** Read-then-write is the one
legitimate multi-step flow this project uses. Treating a workspace
read as external ingest would break it to defend against the
operator's own files.

**Prompt-level defences are nudges.** The provenance markers around
tool output in `router/prompt.py` ask the model not to be steered by
what it reads. They are not what anything rests on; the escalation
guard is.

**`MAX_STEPS=1` is doing real work.** At the default, there is never a
second routing decision, so tool output never reaches a prompt that
chooses a tool. Raising it is what opens that surface, and what the
escalation guard exists for.

**In-memory rate limiting is single-process.** Multiple workers each
keep their own window.

**`sysadmin` can be confidently wrong.** It reads real logs and asks a
local model to explain them. Its output is a proposal for a human,
never applied automatically — by design, no command in
`graphs/sysadmin.py` can mutate anything, and that is fixed in code
rather than configurable.

## Reporting a vulnerability

Open a GitHub issue on
[Kurtisone/forge](https://github.com/Kurtisone/forge/issues), or use
GitHub's private vulnerability reporting for anything you would rather
not post publicly. This is a personal project maintained by one
person: expect a best-effort response, not an SLA.
