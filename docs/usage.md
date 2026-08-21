# Usage

Running Forge: the REPL, the web UI, the CLI, and the container.

## Usage

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
  -v ${XDG_RUNTIME_DIR}/forge-podman-ro-proxy.sock:/run/forge-podman-ro-proxy.sock:ro \
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
SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy.sock
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

REPL commands: `!help`, `!clear`, `!trace`, `!remember`, `!recall`. Multi-line paste: type your question
then append ` ``` ` or paste question + code in one go (auto-detected via `select()`).

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
