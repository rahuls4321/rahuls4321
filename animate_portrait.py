#!/usr/bin/env python3
"""Make a stationary, subtle portrait animation from the supplied photograph.

The animation deliberately reuses the original image pixels rather than replacing
its face with a generated one. Coordinates below refer to the 160 x 200 reference.

Install: python -m pip install numpy pillow opencv-python-headless imageio-ffmpeg
Run:     python animate_portrait.py [input.jpg] [output.mp4]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

FPS = 30
DURATION = 5
# Keep the photograph's 4:5 composition. This is an upscale, NOT extra detail
# reconstructed from the very small reference photograph.
OUTPUT_SIZE = (2160, 2700)


def smoothstep(a: float, b: float, x: float) -> float:
    v = np.clip((x - a) / (b - a), 0.0, 1.0)
    return float(v * v * (3.0 - 2.0 * v))


def blink_amount(t: float, start: float) -> float:
    # 90 ms closing, 45 ms closed, 120 ms opening: natural quick blinks.
    close = smoothstep(start, start + 0.09, t)
    open_ = 1.0 - smoothstep(start + 0.135, start + 0.255, t)
    return close * open_


def build_smile_field(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse displacement field: lip corners lift and the cheeks follow slightly."""
    y, x = np.mgrid[:height, :width].astype(np.float32)

    def blob(cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
        return np.exp(-0.5 * (((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))

    left = blob(58.5, 109.5, 11.0, 7.3)
    right = blob(101.5, 109.5, 11.0, 7.3)
    left_cheek = blob(46.0, 101.0, 15.0, 11.5)
    right_cheek = blob(114.0, 101.0, 15.0, 11.5)
    lower_lip = blob(80.0, 114.0, 18.0, 4.0)

    # Max corner lift is about three original pixels, or <4% of face width.
    # Sign is reversed here because cv2.remap needs destination-to-source maps.
    dx = -2.2 * left + 2.2 * right - 0.28 * left_cheek + 0.28 * right_cheek
    dy = -2.75 * (left + right) - 0.72 * (left_cheek + right_cheek) - 0.43 * lower_lip

    # Exactly pin the nose, upper face, jawline, neck, hair and background.
    yy = y[:, 0]
    ramp_top = np.clip((yy - 84.0) / 13.0, 0.0, 1.0)
    ramp_bottom = np.clip((143.0 - yy) / 19.0, 0.0, 1.0)
    envelope = (ramp_top * ramp_top * (3 - 2 * ramp_top) *
                ramp_bottom * ramp_bottom * (3 - 2 * ramp_bottom))[:, None]
    return dx * envelope, dy * envelope


def closed_eyes(image: np.ndarray, keyframe_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a blink keyframe made from this photo with only the lids changed.

    The keyframe's small closed-eyelid patches were color-matched to the
    original photo. Every other pixel is from the original photograph.
    """
    height, width = image.shape[:2]
    closed = cv2.imread(str(keyframe_path), cv2.IMREAD_COLOR)
    if closed is None or closed.shape != image.shape:
        raise ValueError(f"Blink keyframe must match source dimensions: {keyframe_path}")

    # Feather only the eyelids, leaving the brows, nose and skin untouched.
    eye_alpha = np.zeros((height, width), np.float32)
    for cx in (59, 100):
        layer = np.zeros((height, width), np.uint8)
        cv2.ellipse(layer, (cx, 67), (14, 7), 0, 0, 360, 255, -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (5, 5), 1.0)
        eye_alpha = np.maximum(eye_alpha, layer.astype(np.float32) / 255.0)
    return closed, eye_alpha


def render(source: Path, destination: Path, size: tuple[int, int] = OUTPUT_SIZE,
           keyframe_path: Path = Path("blink_keyframe.png")) -> None:
    original = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not read {source}")
    # The exact input frame is retained; no face swapping, resynthesis or crop.
    height, width = original.shape[:2]
    if (width, height) != (160, 200):
        # Coordinates are normalized for the supplied 160x200 photograph.
        original = cv2.resize(original, (160, 200), interpolation=cv2.INTER_AREA)
        height, width = original.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    delta_x, delta_y = build_smile_field(height, width)
    closed, eye_alpha = closed_eyes(original, keyframe_path)

    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(FPS), "-i", "-", "-an",
        "-vf", f"scale={size[0]}:{size[1]}:flags=lanczos,format=yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
        "-movflags", "+faststart", "-r", str(FPS), str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for i in range(FPS * DURATION):
            t = i / FPS
            smile = smoothstep(0.52, 3.9, t)
            blink = max(blink_amount(t, 1.30), blink_amount(t, 3.36))

            if smile == 0:
                frame = original.copy()
            else:
                frame = cv2.remap(
                    original, xx - smile * delta_x, yy - smile * delta_y,
                    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101,
                )
                # Remapping identity pixels is not always bit-perfect. Put the
                # untouched image back outside the moving facial areas.
                frame[:84] = original[:84]
                frame[143:] = original[143:]

            if blink > 0:
                weight = (blink * eye_alpha)[:, :, None]
                frame = np.clip(frame * (1.0 - weight) + closed * weight, 0, 255).astype(np.uint8)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg could not encode the animation: {error}")
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    print(f"Wrote {destination}: {size[0]}x{size[1]}, {FPS} fps, {DURATION} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("reference_photo.jpg"))
    parser.add_argument("destination", nargs="?", type=Path, default=Path("portrait_animation.mp4"))
    parser.add_argument("--width", type=int, default=OUTPUT_SIZE[0])
    parser.add_argument("--height", type=int, default=OUTPUT_SIZE[1])
    parser.add_argument("--blink-keyframe", type=Path, default=Path("blink_keyframe.png"))
    args = parser.parse_args()
    render(args.source, args.destination, (args.width, args.height), args.blink_keyframe)
