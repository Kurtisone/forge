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

### 2. systemd units -- forge-dbus-proxy (xdg-dbus-proxy) + busctl

Discovery talks to systemd over D-Bus -- no file-based equivalent
exists. `forge-dbus-proxy.sh` uses `xdg-dbus-proxy` (the same tool
Flatpak uses to sandbox D-Bus access) to expose a NEW socket that only
forwards five read-only method calls and silently drops everything
else -- `StartUnit`/`StopUnit`/`RestartUnit`/`Reboot`/... never reach
the real bus at all, confirmed in practice: `ListUnits` succeeds,
a mutating call returns `Access denied` straight from the proxy.

```
-v ${XDG_RUNTIME_DIR}/forge-dbus-proxy:/run/forge-dbus-proxy:ro
```

```
SYSADMIN_DBUS_ADDRESS=unix:path=/run/forge-dbus-proxy/bus
```

**`systemctl` was tried first and abandoned** -- confirmed with
`SYSTEMD_LOG_LEVEL=debug` in production that it does NOT use
`DBUS_SYSTEM_BUS_ADDRESS` at all for this: it hardcodes an attempt at
`/run/systemd/private` (a systemd-specific shortcut protocol to PID 1,
not standard D-Bus) and gives up immediately if that exact path
doesn't exist, with zero fallback. Mounting our proxy's socket
directly onto `/run/systemd/private` was tried next and also failed
(`No data available` -- a real D-Bus client doesn't speak that
shortcut protocol's handshake). `graphs/sysadmin.py` instead uses
`busctl --json=short --address=<proxy> call ...` for discovery --
`busctl` is a generic D-Bus client with none of `systemctl`'s
container-detection quirks, and it was the one tool that worked
consistently against the proxy throughout this whole investigation.
`journalctl`/`podman logs` for actually reading logs are unaffected by
any of this (neither goes through D-Bus).

### 3. podman logs/ps -- podman_ro_proxy.py

podman.sock exposes podman's full REST API. A `:ro` bind mount only
protects the socket *file*, not the *requests* sent through it.
`podman_ro_proxy.py` (stdlib-only) forwards only
`GET /containers/json` and `GET /containers/{id}/logs`, and returns
403 on every other verb or path -- confirmed in practice: a `POST
/containers/create` gets `403 Forbidden` from the proxy itself. See
`tests/test_podman_ro_proxy.py` for the filter's own test coverage.

```
SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy/sock
```

## Running Forge with both proxies mounted

**Group access is required** for `journalctl -u <unit>` on root-owned
system services (see "Group access" below) -- without it, `sysadmin`
still works, but only sees generic recent-entries queries and units
logging to the user session journal, not root services like
`steamos-manager.service`.

### podman run

```bash
podman run -d --name forge \
  --userns=keep-id \
  --group-add keep-groups \
  --env-file .env.local \
  -v $(pwd)/data:/app/data \
  -v /var/log/journal:/host-journal:ro \
  -v ${XDG_RUNTIME_DIR}/forge-dbus-proxy:/run/forge-dbus-proxy:ro \
  -v ${XDG_RUNTIME_DIR}/forge-podman-ro-proxy:/run/forge-podman-ro-proxy:ro \
  -p 8000:8000 \
  forge-core
```

### podman-compose

Same effect, `--group-add keep-groups`'s underlying crun annotation
spelled out directly (works the same way in compose, which has no
`--group-add` flag of its own):

```yaml
services:
  forge:
    image: forge-core
    container_name: forge
    restart: unless-stopped
    userns_mode: "keep-id"
    annotations:
      run.oci.keep_original_groups: "1"
    ports:
      - "8000:8000"
    env_file:
      - .env.local
    volumes:
      - ./data:/app/data
      - /var/log/journal:/host-journal:ro
      - ${XDG_RUNTIME_DIR}/forge-dbus-proxy:/run/forge-dbus-proxy:ro
      - ${XDG_RUNTIME_DIR}/forge-podman-ro-proxy:/run/forge-podman-ro-proxy:ro
```

(`${XDG_RUNTIME_DIR}` on the host side -- e.g. `/run/user/1000` on a
rootless setup; the paths on the right of each `:` are what
`SYSADMIN_JOURNAL_DIR`/`SYSADMIN_DBUS_ADDRESS`/`SYSADMIN_PODMAN_URL`
in `.env.local` above actually refer to, since those are seen from
*inside* the container.)

## Running non-root (and why `--userns=keep-id` is not optional)

The image sets `USER forge` (UID/GID 1000) so the serving process
can't rewrite `/app/src/forge/` or anything else in the container.
That half is in the Containerfile. The other half is at runtime, and
**an image built non-root without `--userns=keep-id` is a broken
container, not a hardened one.**

Why: under rootless podman the container's UID 0 is already mapped to
your host UID. That mapping is what lets the container read the two
proxy sockets (mode 0660, owned by your host user) and write to
`./data`. A container UID of 1000 maps by default to a *subuid*
instead -- a stranger to all of those -- so `sysadmin` loses both
proxies and Forge loses its writable directory. `--userns=keep-id`
maps your host UID to the same UID inside, putting `forge` back where
container-root used to be.

Check your host UID first:

```bash
id -u        # expected: 1000
```

If it isn't 1000, use the explicit form -- `--userns=keep-id:uid=1000,gid=1000`
on the CLI, or `userns_mode: "keep-id:uid=1000,gid=1000"` in compose
-- which pins the mapping to the image's user regardless of the host
id.

### What to verify after switching

Each of these has a distinct failure mode, so run them in order and
stop at the first surprise:

```bash
# 1. Non-root, and mapped to your host user.
podman exec forge id
#    expect: uid=1000(forge) gid=1000(forge)

# 2. Supplementary groups survived the user namespace.
#    This is the one to watch: --group-add keep-groups and
#    --userns=keep-id are two different mechanisms touching the same
#    thing, and keep-groups is what journalctl -u depends on.
podman exec forge id
#    expect: extra groups listed, not just 1000(forge)

# 3. The writable path is actually writable.
podman exec forge touch /app/data/.write-test && echo OK

# 4. Both proxies still reachable. Use a *root-owned* unit here:
#    it's the one subject to the wheel ACL on system.journal, so it
#    tests group access rather than just "no entries".
podman exec forge podman --url unix:///run/forge-podman-ro-proxy/sock ps
podman exec forge journalctl -D /host-journal -u steamos-manager.service -n 5
```

Confirmed working on the Deck (2026-08-13): `uid_map` shows
`1000 0 1`, which is keep-id's signature -- container UID 1000 is the
host user. Supplementary groups survive the switch, because the
kernel evaluates the journal ACL against the host-side GIDs, not the
names visible inside the container.

### If something breaks and you need the old behaviour now

`user: "0:0"` in compose (or `--user 0:0` on the CLI) overrides the
image's `USER` without a rebuild. That is the whole revert: one line,
no image work, and it restores exactly the previous posture.

## Group access for `journalctl -u` on root-owned system services

Confirmed root cause: rootless podman does NOT pass the host user's
supplementary groups (`wheel`, `adm`, etc.) into the container by
default -- `podman exec forge id` showed `groups=0(root)` even though
`deck` on the host is in `wheel`, which is exactly what the
`system.journal` file's ACL requires for read access
(`getfacl` showed `group:wheel:r-x`, nothing for "other"). Generic
`journalctl -D ... -n N` queries and `podman logs` still worked
without this because they don't depend on that ACL the same way; only
`-u <unit>` on root-owned services needs it.

Fix: `crun` (the default OCI runtime) can skip the `setgroups()` call
that normally strips supplementary groups on container entry --
`--group-add keep-groups` on the CLI, or the equivalent annotation
`run.oci.keep_original_groups: "1"` in compose (no `--group-add` flag
exists there). Confirmed in production: `podman exec forge id` then
shows real supplementary groups instead of just `root`, and
`journalctl -D /host-journal -u steamos-manager.service` returns real
entries.

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

- **`systemctl ... Failed to connect to system scope bus via local
  transport: No such file or directory`** persisting even with the
  proxy confirmed working via `busctl` directly (same address, same
  socket) -- `systemctl` never even tries `DBUS_SYSTEM_BUS_ADDRESS`
  for this: `SYSTEMD_LOG_LEVEL=debug` shows it hardcodes an attempt at
  `/run/systemd/private` first and gives up immediately if that exact
  path doesn't exist, with no fallback to the env var at all.
  Bind-mounting the proxy's socket onto that exact path was tried next
  and produced a *different* error (`No data available`) -- that path
  uses a systemd-specific shortcut protocol, not standard D-Bus, so a
  generic D-Bus proxy can't speak it correctly either. The actual fix
  was to stop using `systemctl` for discovery entirely and use
  `busctl` instead (see the section above) -- it has none of these
  quirks and was the one tool that worked consistently against the
  proxy throughout this whole investigation.

- **`journalctl -D ... -u <unit>` returns nothing for a root-owned
  system service** (e.g. `steamos-manager.service`), even though
  generic `-n N` queries and `podman logs` both work fine through the
  same mount -- `podman exec forge id` showing `groups=0(root)` is
  the tell: rootless podman doesn't pass host supplementary groups
  (`wheel`, `adm`, ...) into the container by default, and
  `system.journal`'s ACL requires one of those groups. Fix: see
  "Group access for `journalctl -u`" above (`--group-add keep-groups`
  or the compose annotation).


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
