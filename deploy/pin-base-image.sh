#!/usr/bin/env bash
#
# Resolve the Containerfile's base image tag to an immutable digest
# and (with --write) pin the FROM line to it. Audit E-4.
#
# Why this is a script and not just a digest committed by hand: a
# digest is only correct for the moment it was resolved, and the
# whole point of pinning is that bumping it is a deliberate, visible
# act. `python:3.12` is a moving tag -- it gets rebuilt for every
# CPython patch release and every Debian security update, so two
# builds of the same Forge commit can sit on two different base
# images. That's the same problem the requirements pins fixed, one
# layer down.
#
# Usage:
#   ./deploy/pin-base-image.sh            # show the digest, change nothing
#   ./deploy/pin-base-image.sh --write    # rewrite the FROM line
#
# To bump later: put the tag back (or edit the tag in TAG below),
# re-run with --write, and commit the one-line diff. The diff is the
# record of what you moved onto and when.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repo root, regardless of cwd

TAG="${FORGE_BASE_TAG:-docker.io/library/python:3.12}"
WRITE=0
[[ "${1:-}" == "--write" ]] && WRITE=1

if ! command -v podman >/dev/null 2>&1; then
  echo "podman not found -- this needs to run on the build host." >&2
  exit 1
fi

echo "Pulling ${TAG} to resolve its digest..."
podman pull "${TAG}" >/dev/null

# RepoDigests is the registry-side content address: pulling by it
# gets the exact same bytes forever, where the tag does not. Take the
# first entry -- an image pulled from one registry has one.
DIGEST_REF="$(podman image inspect --format '{{index .RepoDigests 0}}' "${TAG}")"

if [[ -z "${DIGEST_REF}" ]]; then
  echo "No RepoDigest on ${TAG} -- was it built locally rather than pulled?" >&2
  exit 1
fi

echo
echo "Resolved: ${DIGEST_REF}"
echo

if [[ "${WRITE}" -eq 0 ]]; then
  echo "Nothing written. Re-run with --write to pin the Containerfile:"
  echo "  FROM ${DIGEST_REF}"
  exit 0
fi

# Matches both the tag form and an already-pinned digest form, so
# re-running to bump works the same as running it the first time.
python3 - "$DIGEST_REF" <<'PY'
import re
import sys
from pathlib import Path

digest_ref = sys.argv[1]
path = Path("Containerfile")
text = path.read_text(encoding="utf-8")
new_text, count = re.subn(
    r"^FROM\s+\S+$", f"FROM {digest_ref}", text, count=1, flags=re.MULTILINE
)
if count != 1:
    sys.exit("Could not find a single FROM line to rewrite -- edit by hand.")
path.write_text(new_text, encoding="utf-8")
print(f"Containerfile FROM pinned to {digest_ref}")
PY

echo
echo "Now review and commit the diff:"
echo "  git diff Containerfile"
