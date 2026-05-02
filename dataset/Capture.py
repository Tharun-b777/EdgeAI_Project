"""
capture.py
----------
Captures images from Raspberry Pi camera using Picamera2 + OpenCV.
Each run saves one image to the output folder with an auto-incremented number.

Usage:
    python capture.py                        # saves to ./captures/
    python capture.py --folder my_folder     # saves to ./my_folder/
    python capture.py --preview              # shows live preview before capture
    python capture.py --count 5              # captures 5 images in one run
    python capture.py --interval 2 --count 5 # captures 5 images, 2 seconds apart
"""

import cv2
import os
import time
import argparse
from pathlib import Path
from picamera2 import Picamera2

# ============================================================
# CONFIG
# ============================================================
DEFAULT_FOLDER   = "captures"
IMAGE_PREFIX     = "img"
IMAGE_EXTENSION  = ".jpg"
JPEG_QUALITY     = 95
RESOLUTION       = (1280, 720)   # width x height
PREVIEW_DURATION = 2             # seconds to show preview before capture


# ============================================================
# HELPERS
# ============================================================

def get_next_index(folder: Path, prefix: str, ext: str) -> int:
    """
    Scans the folder and returns the next available image number.
    If folder has img_001.jpg, img_002.jpg → returns 3.
    """
    existing = list(folder.glob(f"{prefix}_*{ext}"))
    if not existing:
        return 1
    indices = []
    for f in existing:
        stem = f.stem                    # e.g. "img_007"
        part = stem.replace(f"{prefix}_", "")
        if part.isdigit():
            indices.append(int(part))
    return max(indices) + 1 if indices else 1


def save_image(frame, folder: Path, index: int) -> Path:
    """Saves a BGR frame as JPEG and returns the saved path."""
    filename = f"{IMAGE_PREFIX}_{index:03d}{IMAGE_EXTENSION}"
    out_path = folder / filename
    cv2.imwrite(
        str(out_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    return out_path


def show_preview(picam2, duration: float):
    """
    Shows a live OpenCV preview window for `duration` seconds.
    Press 'q' to skip the wait and capture immediately.
    """
    print(f"  Preview window open — capturing in {duration}s (press 'q' to capture now)...")
    start = time.time()
    while time.time() - start < duration:
        frame = picam2.capture_array()
        # Picamera2 returns RGB — convert to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Preview — press 'q' to capture", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Capture images from Raspberry Pi camera."
    )
    parser.add_argument(
        "--folder", type=str, default=DEFAULT_FOLDER,
        help=f"Output folder (default: {DEFAULT_FOLDER})"
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of images to capture in one run (default: 1)"
    )
    parser.add_argument(
        "--interval", type=float, default=0.0,
        help="Seconds between captures when --count > 1 (default: 0)"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Show live preview window before each capture"
    )
    args = parser.parse_args()

    # ── prepare output folder ────────────────────────────────
    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)

    # ── start camera ─────────────────────────────────────────
    print("\nStarting camera...")
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": RESOLUTION, "format": "RGB888"}
        )
    )
    picam2.start()
    time.sleep(1)   # let camera warm up and auto-expose

    print(f"Camera ready  |  Resolution: {RESOLUTION[0]}x{RESOLUTION[1]}")
    print(f"Output folder : {folder.resolve()}")
    print(f"Images to take: {args.count}")
    print("-" * 50)

    # ── capture loop ─────────────────────────────────────────
    captured = 0
    try:
        for i in range(args.count):
            # show preview if requested
            if args.preview:
                show_preview(picam2, PREVIEW_DURATION)

            # get the next index based on what's already in the folder
            idx = get_next_index(folder, IMAGE_PREFIX, IMAGE_EXTENSION)

            # capture frame
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # save
            saved_path = save_image(frame_bgr, folder, idx)
            captured += 1

            print(f"  [{captured}/{args.count}]  Saved → {saved_path.name}"
                  f"  ({frame_bgr.shape[1]}x{frame_bgr.shape[0]})")

            # wait between captures if requested
            if args.interval > 0 and i < args.count - 1:
                print(f"  Waiting {args.interval}s...")
                time.sleep(args.interval)

    finally:
        picam2.stop()
        print("-" * 50)
        print(f"Done. {captured} image(s) saved to: {folder.resolve()}")


if __name__ == "__main__":
    main()