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
systemctl --user enable forge-dbus-proxy.service
systemctl --user enable forge-podman-ro-proxy.service
# `enable --now` is NOT enough here, and that is the whole reason these
# are two lines instead of one.
#
# --now starts a unit that is stopped and does nothing at all to a unit
# that is already running. So on a FIRST install it looks identical to
# a restart, and on every RE-run -- which this script advertises itself
# as safe for -- it leaves the old process alive with the old
# ExecStart. systemd then reports "active" perfectly truthfully about a
# process running arguments that no longer exist in any file on disk.
#
# Hit on 2026-08-22 upgrading the podman proxy's socket path: the unit
# on disk said --listen %t/forge-podman-ro-proxy/sock, the running
# process still said --listen %t/forge-podman-ro-proxy.sock, is-active
# said "active", and the socket was in neither place anyone was
# looking.
systemctl --user restart forge-dbus-proxy.service
systemctl --user restart forge-podman-ro-proxy.service

sleep 1  # give both a moment to create their sockets before checking
echo
echo "Status:"
systemctl --user is-active forge-dbus-proxy.service forge-podman-ro-proxy.service
echo
echo "Sockets:"
# Checked one at a time, with a diagnosis rather than `ls`'s error.
# Under `set -e` a bare `ls` of a missing socket aborted the script
# here -- so the run that most needed the closing instructions was
# exactly the run that never printed them.
missing=0
for sock in \
  "${XDG_RUNTIME_DIR}/forge-dbus-proxy/bus" \
  "${XDG_RUNTIME_DIR}/forge-podman-ro-proxy/sock"
do
  if [ -S "$sock" ]; then
    ls -la "$sock"
  else
    missing=1
    echo "  MISSING: $sock" >&2
  fi
done

if [ "$missing" -eq 1 ]; then
  echo >&2
  echo "One or both proxies are 'active' without their socket. Almost" >&2
  echo "always this means the RUNNING process predates the unit file." >&2
  echo "Compare what systemd would start against what is actually" >&2
  echo "running:" >&2
  echo >&2
  echo "  systemctl --user show -p ExecStart --value \\" >&2
  echo "    forge-podman-ro-proxy.service" >&2
  echo "  ps -p \"\$(systemctl --user show -p MainPID --value \\" >&2
  echo "    forge-podman-ro-proxy.service)\" -o args=" >&2
  echo >&2
  echo "If the two --listen paths differ, that is it. Otherwise:" >&2
  echo "  journalctl --user -u forge-podman-ro-proxy.service -n 40 --no-pager" >&2
  exit 1
fi

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
