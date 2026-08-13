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
FROM docker.io/library/python@sha256:cfdcc988c45d6a933e0ec3fd9ce46e6f78174d3f082eea8f2f4d6f1f72f32b89

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

# --- Non-root runtime user (audit E-4) --------------------------------
# Everything above runs as root because installing packages needs to;
# nothing below does. uvicorn binds 8000, which is unprivileged, and
# Forge's only writable path is /app/data (MEMORY_FILE, TRACE_FILE,
# RAG_DB_FILE and WORKSPACE_DIR all default under `data/`, and /app is
# WORKDIR). So there is no reason for the serving process to be able
# to rewrite /app/src/forge/ -- which is exactly what a `shell` or
# `files` dispatch could do while it was root.
#
# UID/GID 1000 is not arbitrary and not cosmetic. Under rootless
# podman the container's UID 0 is already mapped to your host UID, and
# that mapping is what makes the 0660 proxy sockets and the mounted
# journal readable at all. Dropping to a container UID that maps to a
# *subuid* would silently break every one of those. Running this image
# non-root therefore requires --userns=keep-id at runtime, so that
# container UID 1000 maps back to host UID 1000. See deploy/README.md,
# "Running non-root" -- the image half without the runtime half is a
# broken container, not a safer one.
#
# python:3.12 (Debian) ships no user at 1000, so this claims it
# cleanly; useradd fails the build loudly if that ever stops being
# true rather than quietly picking a different id.
RUN groupadd --gid 1000 forge \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin forge \
    && mkdir -p /app/data \
    && chown -R forge:forge /app/data /home/forge \
    && python -m compileall -q /app/src

# HOME matters here in a way it didn't as root: the podman client
# resolves its config and scratch paths from it, and /root exists
# while /home/forge only does because it was just created.
ENV HOME=/home/forge

USER forge:forge

EXPOSE 8000

# Default: HTTP API (accessible from browser / other machines)
# Override for REPL: podman run -it ... forge-core python -m forge.main
#
# Plain exec form: uvicorn is PID 1 and receives SIGTERM directly from
# `podman stop`, with no shell in between.
#
# This used to be wrapped in `sh -c "mkdir -p /run/systemd/system &&
# ..."`, to satisfy sd_booted() before `systemctl` would attempt a bus
# connection. Nothing calls systemctl any more -- graphs/sysadmin.py
# discovers units with `busctl --json=short`, which has no such check
# -- so the directory served no purpose and the mkdir was doing
# nothing but adding a shell and a failure mode.
CMD ["uvicorn", "forge.api:app", "--host", "0.0.0.0", "--port", "8000"]
