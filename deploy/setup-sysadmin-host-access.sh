#!/usr/bin/env bash
#
# One-shot setup for sysadmin's read-only host access (see
# deploy/README.md for the full design). Idempotent -- safe to re-run.
#
# What this does, in order (each step mirrors a real issue hit
# deploying this the first time, see the comments below):
#   1. Checks xdg-dbus-proxy is installed (it's a Flatpak dependency,
#      present by default on most desktop distros incl. SteamOS --
#      this script does NOT install packages itself, since SteamOS's
#      read-only root makes that fragile; see deploy/README.md).
#   2. Enables podman.socket -- NOT active by default even when
#      podman itself works fine, because `podman ps`/`logs` via the
#      CLI don't need the REST API socket at all; only a REST client
#      (like our proxy) does.
#   3. Installs the two systemd --user unit files and enables them,
#      so both proxies persist across reboots/logout instead of being
#      background jobs you have to relaunch by hand every session.
#
# Usage: ./deploy/setup-sysadmin-host-access.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root, regardless of cwd

echo "[1/3] Checking for xdg-dbus-proxy..."
if ! command -v xdg-dbus-proxy >/dev/null 2>&1; then
  echo "  Not found. On SteamOS, do NOT 'pacman -S' this (read-only root," >&2
  echo "  survives badly across system updates) -- see deploy/README.md" >&2
  echo "  for the Distrobox/toolbox alternative. Aborting." >&2
  exit 1
fi
echo "  OK: $(command -v xdg-dbus-proxy)"

echo "[2/3] Enabling podman.socket (needed for the REST API, not just the CLI)..."
systemctl --user enable --now podman.socket
echo "  OK: $(systemctl --user is-active podman.socket)"

echo "[3/3] Installing and enabling forge-dbus-proxy + forge-podman-ro-proxy..."
mkdir -p "$HOME/.config/systemd/user"
# __FORGE_REPO_DIR__ is substituted here rather than hardcoded in the
# unit files themselves -- a hardcoded path (e.g. assuming
# ~/Forge) silently breaks the moment the clone lives anywhere else
# (this exact mistake shipped once already: the units assumed
# %h/Forge, the real clone was at %h/Documents/Forge -- systemd's own
# %h specifier can't help here since it only expands $HOME, not the
# repo's actual location under it).
sed "s|__FORGE_REPO_DIR__|$PWD|g" deploy/systemd/forge-dbus-proxy.service \
  > "$HOME/.config/systemd/user/forge-dbus-proxy.service"
sed "s|__FORGE_REPO_DIR__|$PWD|g" deploy/systemd/forge-podman-ro-proxy.service \
  > "$HOME/.config/systemd/user/forge-podman-ro-proxy.service"
systemctl --user daemon-reload
# Clears systemd's restart-rate-limit lockout (StartLimitBurst) from
# any earlier crash loop -- without this, a fixed unit can still
# refuse to (re)start immediately with "Start request repeated too
# quickly" even though the underlying problem is already gone.
systemctl --user reset-failed forge-dbus-proxy.service forge-podman-ro-proxy.service 2>/dev/null || true
systemctl --user enable --now forge-dbus-proxy.service
systemctl --user enable --now forge-podman-ro-proxy.service

sleep 1  # give both a moment to create their sockets before checking
echo
echo "Status:"
systemctl --user is-active forge-dbus-proxy.service forge-podman-ro-proxy.service
echo
echo "Sockets:"
ls -la "${XDG_RUNTIME_DIR}/forge-dbus-proxy/bus" "${XDG_RUNTIME_DIR}/forge-podman-ro-proxy/sock"
echo
echo "Done. Add to .env.local:"
echo "  SYSADMIN_DBUS_ADDRESS=unix:path=/run/forge-dbus-proxy/bus"
echo "  SYSADMIN_PODMAN_URL=unix:///run/forge-podman-ro-proxy/sock"
echo "Then mount both sockets when running Forge's container -- see README.md's"
echo "'API server' section for the full podman run command."
echo
echo "Journal access (journalctl -u/-k) is separate and not handled by this"
echo "script -- it's just a bind mount, no proxy/service needed. Add"
echo "-v /var/log/journal:/host-journal:ro to the podman run command and"
echo "SYSADMIN_JOURNAL_DIR=/host-journal to .env.local. See deploy/README.md."
