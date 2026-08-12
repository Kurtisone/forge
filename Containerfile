# Audit E-4: this tag is mutable. `python:3.12` is rebuilt for every
# CPython patch release and every Debian security update, so two
# builds of the same Forge commit can land on two different base
# images -- the same drift the requirements.txt pins just closed, one
# layer down. Pin it to a digest with:
#
#     ./deploy/pin-base-image.sh --write
#
# which rewrites this line to FROM ...python:3.12@sha256:<digest> and
# leaves a one-line diff to commit. Re-run it to bump, deliberately,
# with the move visible in git history instead of happening silently
# on the next rebuild.
#
# Left as a tag here rather than a digest baked in blind: a digest is
# only meaningful if it's one you resolved yourself, on your own
# build host, from the registry you actually pull from.
FROM python:3.12

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir -r requirements.txt

# journalctl (from the systemd package -- reads journal files
# directly, no daemon involved) and the podman CLI client, needed by
# graphs/sysadmin.py. Neither runs as a service inside this
# container; both are just client binaries talking to whatever's
# mounted/proxied in -- see deploy/README.md for the read-only
# access design (bind-mounted journal dir, filtered D-Bus proxy,
# read-only podman API proxy).
RUN apt-get update && apt-get install -y --no-install-recommends \
    systemd \
    podman \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

# Default: HTTP API (accessible from browser / other machines)
# Override for REPL: podman run -it ... forge-core python -m forge.main
#
# `mkdir -p /run/systemd/system` -- NOT running real systemd, just
# satisfying its own sd_booted() check (per systemd's own man page:
# "Internally, this function checks whether the directory
# /run/systemd/system/ exists" -- nothing more). Without this,
# `systemctl` inside sysadmin bails out with "System has not been
# booted with systemd... Failed to connect to system scope bus...
# Host is down" *before ever attempting* the bus connection --
# happens regardless of DBUS_SYSTEM_BUS_ADDRESS being set correctly,
# confirmed in production on 2026-08-11 (the filtered D-Bus proxy
# worked fine when tested directly with busctl; systemctl inside the
# container still refused). /run is a fresh tmpfs at container start,
# so this can't be done at build time -- it has to happen here, in
# CMD, every time the container starts. `exec` keeps uvicorn as PID 1
# so it still receives SIGTERM directly from `podman stop`.
CMD ["sh", "-c", "mkdir -p /run/systemd/system && exec uvicorn forge.api:app --host 0.0.0.0 --port 8000"]
