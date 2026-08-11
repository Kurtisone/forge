# Read-only host access for `sysadmin`

`sysadmin` needs to read the HOST's `journalctl`/`systemctl`/`podman`
state, but Forge's own container has no business holding real
mutation privilege over any of them. Each piece below is optional and
independent; `sysadmin` degrades gracefully without any of them
(falls back to kernel-log diagnosis, or reports the specific missing
piece -- see graphs/sysadmin.py's `_discover_node`).

## Quick setup (recommended)

```bash
./deploy/setup-sysadmin-host-access.sh
```

This is idempotent and does everything in one go: checks
`xdg-dbus-proxy` is present, enables `podman.socket`, installs and
enables the two `systemd --user` units so both proxies persist across
reboots (no more relaunching background jobs by hand every session).
It ends by printing the exact `.env.local` lines to add. Read on only
if you want to understand what it's doing, or need to troubleshoot.

Provisioning a new host with Ansible instead? See
`deploy/ansible/sysadmin-proxies.yml` -- same steps, adapted to your
NiPoGi playbook once you get there.

## The three pieces, and why each is shaped the way it is

### 1. journalctl -- bind mount, no proxy needed

`journalctl` reads journal files directly off disk; no daemon, no
socket. Just a bind mount and a binary (the Containerfile already
installs the binary):

```bash
podman run ... -v /var/log/journal:/host-journal:ro forge-core
```

```
SYSADMIN_JOURNAL_DIR=/host-journal
```

### 2. systemctl -- forge-dbus-proxy (xdg-dbus-proxy)

`systemctl list-units`/`GetUnit` talk to systemd over D-Bus -- no
file-based equivalent exists. `forge-dbus-proxy.sh` uses
`xdg-dbus-proxy` (the same tool Flatpak uses to sandbox D-Bus access)
to expose a NEW socket that only forwards five read-only method calls
and silently drops everything else --
`StartUnit`/`StopUnit`/`RestartUnit`/`Reboot`/... never reach the
real bus at all, confirmed in practice: `ListUnits` succeeds,
`StartUnit` returns `Access denied` straight from the proxy.

```
SYSADMIN_DBUS_ADDRESS=unix:path=/run/forge-dbus-proxy/bus
```

### 3. podman logs/ps -- podman_ro_proxy.py

podman.sock exposes podman's full REST API. A `:ro` bind mount only
protects the socket *file*, not the *requests* sent through it.
`podman_ro_proxy.py` (stdlib-only) forwards only
`GET /containers/json` and `GET /containers/{id}/logs`, and returns
403 on every other verb or path -- confirmed in practice: a `POST
/containers/create` gets `403 Forbidden` from the proxy itself. See
`tests/test_podman_ro_proxy.py` for the filter's own test coverage.

```
SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy.sock
```

## Running Forge with both proxies mounted

```bash
podman run -d --name forge \
  --env-file .env.local \
  -v $(pwd)/data:/app/data \
  -v ${XDG_RUNTIME_DIR}/forge-dbus-proxy:/run/forge-dbus-proxy:ro \
  -v ${XDG_RUNTIME_DIR}/forge-podman-ro-proxy.sock:/run/forge-podman-ro-proxy.sock:ro \
  -p 8000:8000 \
  forge-core
```

(`${XDG_RUNTIME_DIR}` on the host side -- e.g. `/run/user/1000` on a
rootless setup; the paths on the right of each `:` are what
`SYSADMIN_DBUS_ADDRESS`/`SYSADMIN_PODMAN_URL` in `.env.local` above
actually refer to, since those are seen from *inside* the container.)

## Troubleshooting -- real issues hit standing this up

These are the actual failures encountered getting this running the
first time, kept here because they're easy to hit again on a fresh
host:

- **`mkdir: /run/forge-dbus-proxy: Permission non accordée`** -- `/run`
  itself is root-owned; a plain user can't create directories there.
  `forge-dbus-proxy.sh` now defaults to `$XDG_RUNTIME_DIR` (rootless-
  friendly) instead of `/run` directly, so this shouldn't reoccur --
  if it does, you're likely running as a user without a proper
  `XDG_RUNTIME_DIR` set (check `echo $XDG_RUNTIME_DIR`).

- **`busctl ... Failed to connect ... No such file or directory`**
  even with the right flags -- almost always the shell you're running
  `busctl` in doesn't have the env var set (new terminal tab, `export`
  didn't survive). Check `echo $FORGE_DBUS_PROXY_DIR` first.

- **`busctl ... Call failed: org.freedesktop.DBus.Error.ServiceUnknown`**
  on a call that *should* be allowed -- this means the proxy
  connected fine but the `--call=` rule syntax was wrong. The correct
  form is `--call=NAME=RULE` (bus name first, `method@path` second)
  -- easy to get backwards. `forge-dbus-proxy.sh` has the corrected
  syntax; if you hand-edit it, double check against `man
  xdg-dbus-proxy`.

- **`curl ... Empty reply from server`** on `podman_ro_proxy.py`, with
  a `FileNotFoundError` connecting to the upstream socket in the
  proxy's own stderr -- `podman.socket` is very likely inactive.
  Having `podman ps` work from the CLI does NOT mean the REST API
  socket is running; the CLI talks to podman directly for local
  operations and doesn't need it. Fix:
  `systemctl --user enable --now podman.socket`.

- **Both `systemctl --user is-active` calls stuck on `activating`**,
  and `journalctl --user -u forge-podman-ro-proxy.service` shows
  `can't open file '.../deploy/podman_ro_proxy.py': No such file or
  directory` -- the unit files use a repo path substituted at install
  time (`__FORGE_REPO_DIR__`, replaced by
  `setup-sysadmin-host-access.sh` with wherever it's actually run
  from), never a hardcoded guess like `~/Forge` -- a clone living
  anywhere else (`~/Documents/Forge`, `~/projects/forge`, ...) would
  otherwise silently break. If you ever hand-edit or hand-install the
  `.service` files instead of using the setup script, make sure
  `__FORGE_REPO_DIR__` has actually been replaced with a real path.
  After fixing the path, a unit that crash-looped enough times can
  still refuse to restart immediately ("Start request repeated too
  quickly") -- `systemctl --user reset-failed <unit>` clears that
  lockout (the setup script does this automatically on every run).

## Why not just mount the real sockets with a confirmation step in Forge's code?

Because the confirmation would only guard the path Forge's own code
chooses to take -- once a socket is reachable from inside the
container, anything else running in that same container (a future
bug, a compromised dependency, any code execution path at all) can
reach it directly, bypassing the confirmation entirely. These three
proxies make the mutating calls structurally unavailable regardless
of what code runs inside the container -- the same reasoning already
applied to keeping the `git` tool read-only in Forge itself, one
layer further down the stack.
