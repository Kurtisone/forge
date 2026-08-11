# Read-only host access for `sysadmin`

`sysadmin` needs to read the HOST's `journalctl`/`systemctl`/`podman`
state, but Forge's own container has no business holding real
mutation privilege over any of them. This directory has the three
independent pieces that make that possible while staying
structurally read-only -- not "read-only by convention in the Forge
code", but read-only because the mutating calls simply don't exist on
the other end of what's mounted into the container.

Each piece is optional and independent; `sysadmin` degrades
gracefully without any of them (falls back to kernel-log diagnosis,
or reports the specific missing piece -- see graphs/sysadmin.py's
`_discover_node`).

## 1. journalctl -- bind mount, no proxy needed

`journalctl` reads journal files directly off disk; no daemon, no
socket. So this one is just a bind mount and a binary:

```bash
# host
podman run ... \
  -v /var/log/journal:/host-journal:ro \
  forge-core
```

Install the `journalctl` binary in the Forge image (Containerfile
already does this -- see the `apt-get install` line) and set:

```
SYSADMIN_JOURNAL_DIR=/host-journal
```

## 2. systemctl -- forge-dbus-proxy.sh (xdg-dbus-proxy)

`systemctl list-units`/`GetUnit` talk to systemd over D-Bus -- there
is no file-based equivalent. `forge-dbus-proxy.sh` uses
`xdg-dbus-proxy` (the same tool Flatpak uses to sandbox D-Bus access)
to expose a NEW socket that only forwards five read-only method calls
(`ListUnits`, `ListUnitsByPatterns`, `GetUnit`, and the two
`Properties` getters) and silently drops everything else --
`StartUnit`/`StopUnit`/`RestartUnit`/`Reboot`/... never reach the
real bus at all.

Run it on the host (or as a sidecar sharing a volume with Forge),
started before Forge and kept running alongside it -- wrap it in a
systemd unit for anything beyond manual testing:

```bash
./forge-dbus-proxy.sh &
```

Mount the resulting socket directory into Forge's container and set:

```
SYSADMIN_DBUS_ADDRESS=unix:path=/run/forge-dbus-proxy/bus
```

```bash
podman run ... \
  -v /run/forge-dbus-proxy:/run/forge-dbus-proxy:ro \
  forge-core
```

## 3. podman logs/ps -- podman_ro_proxy.py

podman.sock exposes podman's full REST API. A `:ro` bind mount only
protects the socket *file*, not the *requests* sent through it --
there is no read-only mode for the socket itself. `podman_ro_proxy.py`
is a small stdlib-only HTTP proxy that sits in front of the real
`podman.sock`, forwards only `GET /containers/json` and
`GET /containers/{id}/logs`, and returns 403 on every other verb or
path before it ever reaches the real socket. See
`tests/test_podman_ro_proxy.py` for the filter's own test coverage.

Run it on the host:

```bash
python3 deploy/podman_ro_proxy.py \
  --upstream /run/podman/podman.sock \
  --listen /run/forge-podman-ro-proxy.sock
```

Mount its socket into Forge's container and set:

```
SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy.sock
```

```bash
podman run ... \
  -v /run/forge-podman-ro-proxy.sock:/run/forge-podman-ro-proxy.sock:ro \
  forge-core
```

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
