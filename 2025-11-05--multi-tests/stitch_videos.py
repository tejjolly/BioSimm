#!/usr/bin/env python3
import json
import os
import shlex
import shutil
import subprocess

# ============================================================
# USER SETTINGS
# ============================================================

RUN_PHASES = [2]  # [1], [2], or [1, 2]

PHASE1_MODE = '3d'
# PHASE1_MODE = 'CIP'
# PHASE1_MODE = 'TAG'

if PHASE1_MODE == '3d':
    PHASE1_INPUT_VIDEOS = [
        "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87/3d_dye_frames/animation.mp4",
        "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87_r24/3d_dye_frames/animation.mp4",
        "/Volumes/maxone/2026-02-03--mass_balance/g87_r43/3d_dye_frames/animation.mp4",
        "/Volumes/maxone/2026-02-03--mass_balance/g87_r62/3d_dye_frames/animation.mp4",
    ]
    PHASE1_OUTPUT_VIDEO = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/stitched_3d_dye_frames.mp4"


elif PHASE1_MODE == 'CIP':
    PHASE1_INPUT_VIDEOS = [
        "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87/CIP_animation/g87-intensity-norm/animation.mp4",
        "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87_r24/CIP_animation/g87_r24-intensity-norm/animation.mp4",
        "/Volumes/maxone/2026-02-03--mass_balance/g87_r43/CIP_animation/g87_r43-intensity-norm/animation.mp4",
        "/Volumes/maxone/2026-02-03--mass_balance/g87_r62/CIP_animation/g87_r62-intensity-norm/animation.mp4",
    ]
    PHASE1_OUTPUT_VIDEO = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/stitched_CIP.mp4"


elif PHASE1_MODE == 'TAG':
    PHASE1_INPUT_VIDEOS = [
        "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87/TAG/animation/animation.mp4",
        "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/g87_r24/TAG/animation/animation.mp4",
        "/Volumes/maxone/2026-02-03--mass_balance/g87_r43/TAG/animation/animation.mp4",
        "/Volumes/maxone/2026-02-03--mass_balance/g87_r62/TAG/animation/animation.mp4",
    ]
    PHASE1_OUTPUT_VIDEO = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/stitched_TAG.mp4"
else:
    raise ValueError(f"Unsupported PHASE1_MODE: {PHASE1_MODE}")

# Optional second-stage stitch. These are usually outputs already created by phase 1.
# You can run only this stage by setting RUN_PHASES = [2].
PHASE2_INPUT_VIDEOS = [
    "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/stitched_3d_dye_frames.mp4",
    "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/stitched_CIP.mp4",
]
PHASE2_OUTPUT_VIDEO = "/Volumes/biosimm-Tej-Jolly/2026-02-03--mass_balance/stitched_3d_and_CIP.mp4"
PHASE2_LAYOUT = "horizontal"  # allowed: "vertical", "horizontal"

# Crop applied to every input video in each phase, in percentages of original width/height
PHASE1_TRIM_LEFT_PCT   = 0.00
PHASE1_TRIM_RIGHT_PCT  = 0.25
PHASE1_TRIM_TOP_PCT    = 0.00
PHASE1_TRIM_BOTTOM_PCT = 0.00

PHASE2_TRIM_LEFT_PCT   = 0.00
PHASE2_TRIM_RIGHT_PCT  = 0.00
PHASE2_TRIM_TOP_PCT    = 0.00
PHASE2_TRIM_BOTTOM_PCT = 0.00

# Time trimming applied identically to all videos
# Set to None to disable either one
START_TIME_SEC = None
END_TIME_SEC   = None

# Size of each tile when videos need scale/pad geometry normalization.
# If the post-crop input dimensions already match, scale/pad is skipped and
# the native post-crop dimensions are used instead.
TILE_WIDTH  = 960
TILE_HEIGHT = 540

# Output video settings
OUTPUT_FPS = 100
CRF = 23
PRESET = "ultrafast"

# If True, stop when the shortest input ends
STOP_AT_SHORTEST = True

# ============================================================
# HELPERS
# ============================================================

def run_cmd(cmd):
    print("\nRunning command:\n")
    print(" ".join(shlex.quote(x) for x in cmd))
    print()
    subprocess.run(cmd, check=True)

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH.")

def validate_inputs(input_videos, expected_count, job_name):
    if len(input_videos) != expected_count:
        raise ValueError(f"{job_name} expects exactly {expected_count} input videos.")
    for path in input_videos:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input video not found: {path}")

def validate_crop(crop):
    vals = list(crop)
    for v in vals:
        if not (0.0 <= v < 1.0):
            raise ValueError("All trim percentages must be between 0.0 and 1.0.")
    trim_left_pct, trim_right_pct, trim_top_pct, trim_bottom_pct = crop
    if trim_left_pct + trim_right_pct >= 1.0:
        raise ValueError("TRIM_LEFT_PCT + TRIM_RIGHT_PCT must be < 1.0.")
    if trim_top_pct + trim_bottom_pct >= 1.0:
        raise ValueError("TRIM_TOP_PCT + TRIM_BOTTOM_PCT must be < 1.0.")

def crop_needed(crop):
    vals = list(crop)
    return any(v != 0.0 for v in vals)

def all_dimensions_match(dimensions):
    return len(set(dimensions)) == 1

def dimensions_are_yuv420_compatible(dimensions):
    return all(width % 2 == 0 and height % 2 == 0 for width, height in dimensions)

def scale_pad_needed(post_crop_dimensions):
    return (
        not all_dimensions_match(post_crop_dimensions)
        or not dimensions_are_yuv420_compatible(post_crop_dimensions)
    )

def probe_video_dimensions(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in: {path}")
    width = streams[0].get("width")
    height = streams[0].get("height")
    if width is None or height is None:
        raise RuntimeError(f"Could not read video dimensions for: {path}")
    return int(width), int(height)

def probe_input_dimensions(input_videos):
    return [probe_video_dimensions(path) for path in input_videos]

def crop_geometry(width, height, crop):
    trim_left_pct, trim_right_pct, trim_top_pct, trim_bottom_pct = crop
    crop_w = int(width * (1.0 - trim_left_pct - trim_right_pct))
    crop_h = int(height * (1.0 - trim_top_pct - trim_bottom_pct))
    crop_x = int(width * trim_left_pct)
    crop_y = int(height * trim_top_pct)
    return crop_w, crop_h, crop_x, crop_y

def dimensions_after_crop(input_dimensions, crop):
    if not crop_needed(crop):
        return input_dimensions
    return [
        crop_geometry(width, height, crop)[:2]
        for width, height in input_dimensions
    ]

def trim_duration():
    if END_TIME_SEC is None:
        return None
    if START_TIME_SEC is None:
        return END_TIME_SEC

    duration = END_TIME_SEC - START_TIME_SEC
    if duration <= 0:
        raise ValueError("END_TIME_SEC must be greater than START_TIME_SEC.")
    return duration

def build_stack_filter(stack_inputs, layout):
    if layout == "grid4":
        return (
            f"{''.join(stack_inputs)}"
            f"xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:fill=black[vout]"
        )
    if layout == "horizontal2":
        return (
            f"{''.join(stack_inputs)}"
            f"xstack=inputs=2:layout=0_0|w0_0:fill=black[vout]"
        )
    if layout == "vertical2":
        return (
            f"{''.join(stack_inputs)}"
            f"xstack=inputs=2:layout=0_0|0_h0:fill=black[vout]"
        )
    raise ValueError(f"Unsupported layout: {layout}")

def build_filter_complex(input_dimensions, crop, layout):
    input_dimensions = list(input_dimensions)
    use_crop = crop_needed(crop)
    post_crop_dimensions = dimensions_after_crop(input_dimensions, crop)
    use_scale_pad = scale_pad_needed(post_crop_dimensions)

    parts = []
    stack_inputs = []

    for i, (width, height) in enumerate(input_dimensions):
        filters = []

        if use_crop:
            crop_w, crop_h, crop_x, crop_y = crop_geometry(width, height, crop)
            filters.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")

        if use_scale_pad:
            filters.append(
                f"scale={TILE_WIDTH}:{TILE_HEIGHT}:force_original_aspect_ratio=decrease"
            )
            filters.append(
                f"pad={TILE_WIDTH}:{TILE_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"
            )

        if filters:
            filters.append("setsar=1")
            parts.append(f"[{i}:v]{','.join(filters)}[v{i}]")
            stack_inputs.append(f"[v{i}]")
        else:
            stack_inputs.append(f"[{i}:v]")

    parts.append(build_stack_filter(stack_inputs, layout))

    return ";".join(parts)

def build_ffmpeg_cmd(input_videos, output_video, input_dimensions, crop, layout):
    cmd = ["ffmpeg", "-y"]
    duration = trim_duration()

    for path in input_videos:
        if START_TIME_SEC is not None:
            cmd += ["-ss", str(START_TIME_SEC)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-i", path]

    filter_complex = build_filter_complex(input_dimensions, crop, layout)
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]"]

    if OUTPUT_FPS is not None:
        cmd += ["-r", str(OUTPUT_FPS)]

    cmd += [
        "-c:v", "libx264",
        "-crf", str(CRF),
        "-preset", PRESET,
        "-pix_fmt", "yuv420p",
    ]

    if STOP_AT_SHORTEST:
        cmd += ["-shortest"]

    cmd += [output_video]
    return cmd

def print_geometry_plan(input_videos, input_dimensions, crop):
    post_crop_dimensions = dimensions_after_crop(input_dimensions, crop)
    use_crop = crop_needed(crop)
    use_scale_pad = scale_pad_needed(post_crop_dimensions)

    print("Input dimensions:")
    for path, (width, height) in zip(input_videos, input_dimensions):
        print(f"  {os.path.basename(path)}: {width}x{height}")
    print(f"Crop filter: {'enabled' if use_crop else 'skipped'}")
    print(f"Scale/pad filters: {'enabled' if use_scale_pad else 'skipped'}")
    if use_scale_pad:
        print(f"Normalized tile size: {TILE_WIDTH}x{TILE_HEIGHT}")
    else:
        width, height = post_crop_dimensions[0]
        print(f"Native tile size: {width}x{height}")

# ============================================================
# MAIN
# ============================================================

def phase1_crop():
    return (
        PHASE1_TRIM_LEFT_PCT,
        PHASE1_TRIM_RIGHT_PCT,
        PHASE1_TRIM_TOP_PCT,
        PHASE1_TRIM_BOTTOM_PCT,
    )

def phase2_crop():
    return (
        PHASE2_TRIM_LEFT_PCT,
        PHASE2_TRIM_RIGHT_PCT,
        PHASE2_TRIM_TOP_PCT,
        PHASE2_TRIM_BOTTOM_PCT,
    )

def phase2_layout():
    if PHASE2_LAYOUT == "horizontal":
        return "horizontal2"
    if PHASE2_LAYOUT == "vertical":
        return "vertical2"
    raise ValueError("PHASE2_LAYOUT must be 'vertical' or 'horizontal'.")

def run_stitch_job(job_name, input_videos, output_video, expected_count, crop, layout):
    print(f"\n========== {job_name} ==========")
    validate_inputs(input_videos, expected_count, job_name)
    validate_crop(crop)
    input_dimensions = probe_input_dimensions(input_videos)
    print_geometry_plan(input_videos, input_dimensions, crop)

    out_dir = os.path.dirname(output_video)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = build_ffmpeg_cmd(input_videos, output_video, input_dimensions, crop, layout)
    run_cmd(cmd)

    print(f"Done.\nOutput written to:\n{output_video}")

def main():
    ensure_ffmpeg()
    if 1 in RUN_PHASES:
        run_stitch_job(
            f"Phase 1 ({PHASE1_MODE})",
            PHASE1_INPUT_VIDEOS,
            PHASE1_OUTPUT_VIDEO,
            4,
            phase1_crop(),
            "grid4",
        )
    if 2 in RUN_PHASES:
        run_stitch_job(
            "Phase 2",
            PHASE2_INPUT_VIDEOS,
            PHASE2_OUTPUT_VIDEO,
            2,
            phase2_crop(),
            phase2_layout(),
        )

if __name__ == "__main__":
    main()
