#!/usr/bin/env bash
#
# Run one bench harness inside the Forge container, against a copy of
# the real store, using the code in THIS checkout.
#
#     bench/in_container.sh recall_distance --no-plant \
#         --hit "Quel processeur a mon NiPoGi ?" --expect 308 \
#         --miss "Comment s'appelle mon chat ?"
#
#     bench/in_container.sh instruct_prefix --db /tmp/real_copy.db \
#         --hit "Quel processeur a mon NiPoGi ?" --expect 308 \
#         --miss "Comment s'appelle mon chat ?"
#
# Everything after the harness name is passed straight through, so the
# harnesses keep their own arguments and their own --help.
#
# WHY THIS IS A FILE AND NOT A COMMENT
#
# The procedure was six commands duplicated across the top of three
# harnesses, and every fault in it is silent. `rm -rf` left out means
# podman cp merges into the previous run and the container holds a mix
# of two checkouts. A forgotten `podman cp bench/_harness.py` is the
# one loud failure of the lot. And the database argument is the one
# that matters: point a harness at /app/data/forge_rag.db and it is
# measuring production.
#
# PYTHONPATH IS THE POINT, NOT DECORATION
#
# Copying src/ into the container did nothing on its own. Running
# `python /tmp/arm/recall_distance.py` puts /tmp/arm on sys.path, not
# /tmp/arm/src, so `from forge import rag` resolved through the
# image's own PYTHONPATH=/app/src -- the DEPLOYED code, not the
# checkout that was just copied. The harness file was current and
# everything it called was whatever was last built. Setting
# PYTHONPATH here makes the copy actually win, which is what copying
# it was for.
#
# It matters most in exactly the case you reach for a harness: a
# change to rag.py that has not been rebuilt into the image yet. The
# alternative is to `podman compose build forge` before every
# measurement and remember why.
set -euo pipefail

CONTAINER="${FORGE_CONTAINER:-forge}"
WORKDIR=/tmp/arm
STORE_COPY=/tmp/real_copy.db

if [ $# -lt 1 ]; then
    echo "usage: $0 <harness> [args...]" >&2
    echo "harnesses: $(cd "$(dirname "$0")" && ls *.py | sed 's/\.py//' | grep -v '^_' | tr '\n' ' ')" >&2
    exit 1
fi

HARNESS="$1"
shift

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$BENCH_DIR")"

if [ ! -f "$BENCH_DIR/$HARNESS.py" ]; then
    echo "no such harness: bench/$HARNESS.py" >&2
    exit 1
fi

# Fresh directory every time. Without the rm -rf, podman cp merges
# into whatever the last run left behind and the measurement runs on
# a mix of two checkouts.
podman exec "$CONTAINER" sh -c "rm -rf $WORKDIR && mkdir -p $WORKDIR"
podman cp "$REPO_DIR/src" "$CONTAINER:$WORKDIR/"
podman cp "$BENCH_DIR/$HARNESS.py" "$CONTAINER:$WORKDIR/"
podman cp "$BENCH_DIR/_harness.py" "$CONTAINER:$WORKDIR/"

# A copy, always, even for the harnesses that plant their own
# fixtures and never read this one. Handing a benchmark the real
# store is a one-keystroke mistake with no undo, and the habit is
# what keeps it out of production.
podman exec "$CONTAINER" cp /app/data/forge_rag.db "$STORE_COPY"

echo "--- $HARNESS, against $STORE_COPY, with $WORKDIR/src on the path"
podman exec -e "PYTHONPATH=$WORKDIR/src" -it "$CONTAINER" \
    python "$WORKDIR/$HARNESS.py" "$@"
