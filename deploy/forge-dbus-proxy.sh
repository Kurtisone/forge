#!/usr/bin/env bash
#
# Launches a filtered D-Bus proxy for sysadmin's systemctl access,
# using xdg-dbus-proxy -- the tool Flatpak itself uses to give
# sandboxed processes scoped D-Bus access. Deliberately NOT a
# hand-written dbus policy.conf: raw dbus policy files are host-wide
# (scoped by uid/gid connecting to the bus), not scoped to "just this
# one container" -- xdg-dbus-proxy instead sits in front of the real
# bus and exposes a NEW socket that only forwards the calls explicitly
# allowed below, denying everything else at the proxy itself. This
# runs on the HOST (or as a sidecar container sharing a volume with
# Forge), never inside Forge's own container.
#
# Install (Debian/Ubuntu): apt-get install xdg-dbus-proxy
#
# Usage: run this before starting the Forge container, then mount
# $PROXY_SOCKET_DIR into Forge's container and set in .env.local:
#   SYSADMIN_DBUS_ADDRESS=unix:path=/run/forge-dbus-proxy/bus
#
# For a permanent setup, wrap this in a systemd unit
# (forge-dbus-proxy.service) with Restart=always, started before
# forge.service and stopped after it.

set -euo pipefail

PROXY_SOCKET_DIR="${FORGE_DBUS_PROXY_DIR:-/run/forge-dbus-proxy}"
PROXY_SOCKET="$PROXY_SOCKET_DIR/bus"

mkdir -p "$PROXY_SOCKET_DIR"
rm -f "$PROXY_SOCKET"

exec xdg-dbus-proxy \
  "unix:path=/run/dbus/system_bus_socket" \
  "$PROXY_SOCKET" \
  --filter \
  --call="org.freedesktop.systemd1=org.freedesktop.systemd1.Manager.ListUnits@/org/freedesktop/systemd1" \
  --call="org.freedesktop.systemd1=org.freedesktop.systemd1.Manager.ListUnitsByPatterns@/org/freedesktop/systemd1" \
  --call="org.freedesktop.systemd1=org.freedesktop.systemd1.Manager.GetUnit@/org/freedesktop/systemd1" \
  --call="org.freedesktop.systemd1=org.freedesktop.DBus.Properties.Get@/org/freedesktop/systemd1" \
  --call="org.freedesktop.systemd1=org.freedesktop.DBus.Properties.GetAll@/org/freedesktop/systemd1"

# Note what's deliberately absent: StartUnit, StopUnit, RestartUnit,
# KillUnit, ReloadUnit, EnableUnitFiles, DisableUnitFiles, Reboot,
# PowerOff, and everything else on org.freedesktop.systemd1.Manager.
# --filter makes this a deny-by-default proxy: only the five calls
# listed above ever reach the real bus. Nothing added here should
# ever be a call whose name doesn't start with a read-only verb
# (List/Get) -- if a future need requires more, each addition should
# be reviewed on that basis specifically.
