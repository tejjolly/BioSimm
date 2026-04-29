#!/bin/bash
set -euo pipefail

FPS=100
OUT="animation.mp4"
T0=5.21

ffmpeg -y \
  -framerate "$FPS" \
  -pattern_type glob \
  -i "tag_80_frame_*.png" \
  -vf "drawtext=text='t=%{pts\\:hms\\:$T0}':x=1450:y=1150:fontsize=100:fontcolor=black:box=0" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUT"

echo "Wrote $OUT"