#!/usr/bin/env bash
set -euo pipefail

WORKING_DIR="/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing"
MIRROR_DIR="/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing_mirror.git"
VIEW_DIR="/Users/tejjolly/Documents/BioSimm/Simulations/Post_Processing_mirror_view"
STAMP_FILE="$VIEW_DIR/MIRROR_SYNC.txt"
EXCLUDE_FILE="$VIEW_DIR/.git/info/exclude"

if [ ! -d "$WORKING_DIR/.git" ]; then
  echo "Working repo not found: $WORKING_DIR" >&2
  exit 1
fi

if [ ! -d "$MIRROR_DIR" ]; then
  echo "Mirror repo not found: $MIRROR_DIR" >&2
  exit 1
fi

if [ ! -d "$VIEW_DIR/.git" ]; then
  echo "Mirror view clone not found: $VIEW_DIR" >&2
  exit 1
fi

if ! git -C "$WORKING_DIR" remote | grep -qx "localmirror"; then
  echo "Remote 'localmirror' is not configured in $WORKING_DIR" >&2
  exit 1
fi

if ! grep -qxF "MIRROR_SYNC.txt" "$EXCLUDE_FILE" 2>/dev/null; then
  printf "\n# Local mirror metadata\nMIRROR_SYNC.txt\n" >> "$EXCLUDE_FILE"
fi

git -C "$WORKING_DIR" push --mirror localmirror
git -C "$VIEW_DIR" pull --ff-only

printf "Mirror source: %s\nLast synced: %s\n" \
  "$MIRROR_DIR" \
  "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$STAMP_FILE"

echo "Mirror view refreshed: $STAMP_FILE"
