#!/bin/bash
set -euo pipefail

FPS=100
OUT="animation.mp4"

ffmpeg -y \
  -framerate "$FPS" \
  -pattern_type glob \
  -i "frame_*.png" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "Wrote $OUT"