# Usage

Running Forge: the REPL, the web UI, the CLI, and the container.

## Getting it running

```bash
cp .env.example .env.local   # then edit if you need to override any default
```

`podman build` below picks up the `Containerfile` in the repo root automatically
(podman's native name — no `-f` flag needed). It defaults to serving the API.

**Container networking:** the default LLM backends (llama.cpp on `:8080`,
Ollama on `:11434`) are meant to run on the **host**, not inside the
container. From inside a container, `127.0.0.1` means the container itself.
Point `LLAMA_CPP_URL`/`OLLAMA_URL` in `.env.local` at
`http://host.containers.internal:8080` (podman) instead — already the
convention used by this repo's own `.env.local` setups.

**API server (recommended — accessible from browser and any device on the network):**

```bash
podman build -t forge-core .
podman run -d --name forge \
  --env-file .env.local \
  -v $(pwd)/data:/app/data \
  -p 8000:8000 \
  forge-core

# Open in browser (same machine or any device on the same network)
open http://localhost:8000
open http://<host-ip>:8000
```

Exposing this beyond localhost or a trusted LAN? Set `API_TOKEN` in `.env.local`
first — see [Configuration](configuration.md) and [the HTTP API](api.md).

**Optional: `sysadmin` host access (v3.11)** — mount the read-only proxies
and the journal to let `sysadmin` read the host's real
`journalctl`/`systemctl`/`podman` state instead of falling back to
kernel-only diagnosis:

```bash
./deploy/setup-sysadmin-host-access.sh   # one-time, idempotent

podman run -d --name forge \
  --group-add keep-groups \
  --env-file .env.local \
  -v $(pwd)/data:/app/data \
  -v /var/log/journal:/host-journal:ro \
  -v ${XDG_RUNTIME_DIR}/forge-dbus-proxy:/run/forge-dbus-proxy:ro \
  -v ${XDG_RUNTIME_DIR}/forge-podman-ro-proxy:/run/forge-podman-ro-proxy:ro \
  -p 8000:8000 \
  forge-core
```

(`--group-add keep-groups` — or the compose annotation
`run.oci.keep_original_groups: "1"` — is what lets `journalctl -u
<unit>` read root-owned system services; without it, `sysadmin` still
works but only sees generic queries and user-session units. See
[`deploy/README.md`](../deploy/README.md#group-access-for-journalctl--u-on-root-owned-system-services)
for why.)

In `.env.local`:

```
SYSADMIN_JOURNAL_DIR=/host-journal
SYSADMIN_DBUS_ADDRESS=unix:path=/run/forge-dbus-proxy/bus
SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy/sock
SYSADMIN_MAX_LOG_LINES=100
```

Full design and troubleshooting: [`deploy/README.md`](../deploy/README.md).

**REPL (interactive terminal, local only):**

```bash
podman run -it --rm \
  --env-file .env.local \
  -v $(pwd)/data:/app/data \
  forge-core python -m forge.main
```

REPL commands: `!help`, `!clear`, `!compact`, `!memory [kind]`, `!forget <id>`,
`!trace`, `!capabilities`, `!remember <kind> <project|-> <content>`, `!recall <query>`,
`!pair`.
Multi-line paste: type your question then append ` ``` ` or paste question + code in
one go (auto-detected via `select()`).

Two of those are the pair for reading and repairing the vector store, and they
exist because `!recall` cannot do it: semantic search needs a question, so it can
never tell you what is *in* there. `!memory` lists entries as stored, with their
ids and the breakdown by kind; `!forget <id>` removes one from both tables.

`!clear` empties the rolling history **and the tiroir** — pinned messages
included, which is a known debt. It is also the cheap way to keep a live trial
out of the store: compaction indexes one entry per exchange, so a trial question
still in the window when the threshold is crossed gets archived, and the next
trial's best match is the previous trial.

**CLI (one-shot commands, no REPL):**

```bash
# Review a file
podman run --rm --env-file .env.local \
  -v $(pwd):/workspace forge-core \
  python -m forge.cli review src/forge/main.py "Que peut-on améliorer ?"

# Review a file and run its tests first (v3.10) -- test output becomes
# primary evidence for the review, not just the code itself
python -m forge.cli review src/forge/graph.py --tests tests/test_graph.py

# Replay a past execution trace
python -m forge.cli replay <run_id>

# What Forge can do right now, and what the active policy is blocking
python -m forge.cli capabilities
```

---

[← Documentation index](README.md) · [← Project README](../README.md)

## Pairing a phone

`!pair` draws a QR code that points the Android app at this Forge. It is the
one command that works in both interfaces and is not in `main.py`'s dispatch
alone: typed in the web UI it is intercepted before the router, next to
delegation's interception and for the same reason — an intercepted turn costs
no LLM call at all. In the REPL it prints the same payload drawn with
characters, since a terminal cannot show a PNG.

It needs `FORGE_PUBLIC_URL`: the address the **phone** uses, which over
WireGuard is not the address you use. That value goes into the QR verbatim and
becomes the app's base URL, so `!pair` refuses a loopback one rather than hand
out a code that builds a client calling the phone itself. It also refuses when
`API_TOKEN` is unset — pairing an open instance would publish it onto the
WireGuard network.

**What the QR carries is not the bearer token.** It holds a single-use token,
valid `PAIRING_TTL_SECONDS` (five minutes by default), which the app exchanges
once at `POST /pair/claim` for the durable one. That distinction is the whole
design: the QR is rendered into a conversation, and everything in the
conversation is written down — `memory.json`, the rolling history that prefixes
every later router prompt, and the vector store. A bearer token drawn there
would be permanently readable by anyone who opens the UI and retrievable by a
recall months later.

So the `!pair` turn is the one turn `run()` does not remember. Reloading the
page makes the QR disappear; `!pair` draws a new one. A photographed code is
worth nothing once the phone has scanned it, and nothing at all after five
minutes.

Revoking a device today means changing `API_TOKEN` and restarting, which
revokes every client at once. There is one user and one token; per-device
tokens would earn their keep the day losing one device must not invalidate the
others. `pairing.claim()` already returns *the token this device should use*
rather than a constant, which is where one would be minted.
