import cv2
import numpy as np
import time
import os
import threading
import psutil
import json

# ============================================================
# CONFIG — edit these paths to match your Pi
# ============================================================
IMAGES = {
    "img_6": {
        "path": "./img_6.jpg.6m38p24m.ingestion-5d8ccfdcf7-8hx5j.jpg",   # ← put your image path here
        "boxes": [
        {"label":"laptop screen","x":0,"y":90,"width":325,"height":176},
        {"label":"laptop screen","x":302.4796573875803,"y":80.6659528907923,"width":337.5203426124197,"height":218.89079229122055},
        {"label":"laptop screen","x":423,"y":255.03640256959315,"width":217,"height":189}
        ]
    },
    "img_7": {
        "path": "./img_27.jpg.6m385no3.ingestion-5d8ccfdcf7-ghxq7.jpg",   # ← put your image path here
        "boxes": [
            {"label":"face","x":199,"y":250,"width":67,"height":74},
            {"label":"face","x":491,"y":289,"width":66,"height":64}
        ]
    },
}

NUM_RUNS   = 50       # iterations per blur type per image
OUTPUT_DIR = "blurred_output"   # folder where blurred images + JSON are saved

# ============================================================
# BLUR IMPLEMENTATIONS
# ============================================================
def blur_gaussian_51(roi):
    return cv2.GaussianBlur(roi, (51, 51), 0)

def blur_gaussian_101(roi):
    return cv2.GaussianBlur(roi, (101, 101), 0)

def blur_gaussian_151(roi):
    return cv2.GaussianBlur(roi, (151, 151), 0)

def blur_gaussian_2pass(roi):
    r = cv2.GaussianBlur(roi, (51, 51), 0)
    return cv2.GaussianBlur(r, (51, 51), 0)

def blur_gaussian_3pass(roi):
    r = cv2.GaussianBlur(roi, (51, 51), 0)
    r = cv2.GaussianBlur(r,   (51, 51), 0)
    return cv2.GaussianBlur(r, (51, 51), 0)

def blur_box_51(roi):
    return cv2.blur(roi, (51, 51))

def blur_box_101(roi):
    return cv2.blur(roi, (101, 101))

def blur_median_21(roi):
    return cv2.medianBlur(roi, 21)

def blur_median_51(roi):
    return cv2.medianBlur(roi, 51)

def _pixelate(roi, block):
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def blur_pixelate_10(roi): return _pixelate(roi, 10)
def blur_pixelate_20(roi): return _pixelate(roi, 20)

# StackBlur is available in OpenCV 4.7+ — safe fallback if not present
try:
    cv2.stackBlur(np.zeros((10, 10, 3), dtype=np.uint8), (5, 5))
    def blur_stack_51(roi):  return cv2.stackBlur(roi, (51, 51))
    def blur_stack_101(roi): return cv2.stackBlur(roi, (101, 101))
    HAS_STACK = True
except AttributeError:
    HAS_STACK = False

BLUR_VARIANTS = [
    ("Gaussian  k=51  1-pass", blur_gaussian_51),
    ("Gaussian  k=101 1-pass", blur_gaussian_101),
    ("Gaussian  k=151 1-pass", blur_gaussian_151),
    ("Gaussian  k=51  2-pass", blur_gaussian_2pass),
    ("Gaussian  k=51  3-pass", blur_gaussian_3pass),
    ("Box       k=51",         blur_box_51),
    ("Box       k=101",        blur_box_101),
    ("Median    k=21",         blur_median_21),
    ("Median    k=51",         blur_median_51),
    ("Pixelate  block=10",     blur_pixelate_10),
    ("Pixelate  block=20",     blur_pixelate_20),
]
if HAS_STACK:
    BLUR_VARIANTS += [
        ("StackBlur k=51",  blur_stack_51),
        ("StackBlur k=101", blur_stack_101),
    ]
else:
    print("Note: StackBlur not available (needs OpenCV 4.7+) — skipping.")

# ============================================================
# APPLY BLUR TO ALL BOXES IN A FRAME
# ============================================================
def apply_all_boxes(frame, boxes, blur_fn):
    out = frame.copy()
    h, w = out.shape[:2]
    for box in boxes:
        x1 = max(0, int(box["x"]))
        y1 = max(0, int(box["y"]))
        x2 = min(w, int(box["x"] + box["width"]))
        y2 = min(h, int(box["y"] + box["height"]))
        roi = out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        out[y1:y2, x1:x2] = blur_fn(roi)
    return out

# ============================================================
# BENCHMARK
# ============================================================
def benchmark(label, frame, boxes, blur_fn, num_runs=NUM_RUNS):
    process    = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / 1e6

    cpu_samples = []
    stop_event  = threading.Event()

    def sample_cpu():
        psutil.cpu_percent(interval=None)   # prime — discard first reading
        while not stop_event.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.05))

    sampler = threading.Thread(target=sample_cpu, daemon=True)
    sampler.start()

    times    = []
    peak_ram = ram_before

    for _ in range(num_runs):
        t0 = time.perf_counter()
        apply_all_boxes(frame, boxes, blur_fn)
        times.append((time.perf_counter() - t0) * 1000)

        cur = process.memory_info().rss / 1e6
        if cur > peak_ram:
            peak_ram = cur

    stop_event.set()
    sampler.join(timeout=0.5)

    mean_ms   = sum(times) / len(times)
    min_ms    = min(times)
    max_ms    = max(times)
    fps       = 1000 / mean_ms
    mean_cpu  = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    ram_delta = peak_ram - ram_before

    print(
        f"  {label:<30}"
        f"  avg={mean_ms:7.2f}ms  min={min_ms:6.2f}  max={max_ms:6.2f}"
        f"  ~{fps:7.1f}FPS"
        f"  CPU={mean_cpu:5.1f}%"
        f"  RAM Δ={ram_delta:+.2f}MB"
    )
    return {
        "label":        label,
        "avg_ms":       round(mean_ms, 3),
        "min_ms":       round(min_ms,  3),
        "max_ms":       round(max_ms,  3),
        "fps":          round(fps,     1),
        "cpu_pct":      round(mean_cpu, 1),
        "ram_delta_mb": round(ram_delta, 2),
    }

# ============================================================
# SAVE BLURRED IMAGE
# ============================================================
def save_blurred(img_key, frame, boxes, blur_fn, label, out_dir):
    blurred  = apply_all_boxes(frame, boxes, blur_fn)
    safe     = label.strip().replace(" ", "_").replace("=", "").replace("/", "")
    out_path = os.path.join(out_dir, f"{img_key}__{safe}.jpg")
    cv2.imwrite(out_path, blurred)
    return out_path

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    vm  = psutil.virtual_memory()
    print("=" * 90)
    print(f"  Raspberry Pi Blur Benchmark")
    print(f"  Cores : {psutil.cpu_count()}   "
          f"Total RAM : {vm.total/1e9:.2f}GB   "
          f"Available : {vm.available/1e9:.2f}GB   "
          f"Runs/variant : {NUM_RUNS}")
    print("=" * 90)

    all_results = {}

    for img_key, meta in IMAGES.items():
        frame = cv2.imread(meta["path"])
        if frame is None:
            print(f"\n[ERROR] Could not load image: {meta['path']}")
            print(f"        Make sure the file exists at that path and retry.")
            continue

        boxes  = meta["boxes"]
        h, w   = frame.shape[:2]

        print(f"\n{'━' * 90}")
        print(f"  Image : {img_key}   path={meta['path']}   "
              f"size={w}x{h}   boxes={len(boxes)}")
        print(f"{'━' * 90}")

        # ── save original image with bounding boxes drawn ─────
        orig_annotated = frame.copy()
        for box in boxes:
            x1 = max(0, int(box["x"]))
            y1 = max(0, int(box["y"]))
            x2 = min(w, int(box["x"] + box["width"]))
            y2 = min(h, int(box["y"] + box["height"]))
            cv2.rectangle(orig_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_annotated, box["label"], (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        orig_path = os.path.join(OUTPUT_DIR, f"{img_key}__original_boxes.jpg")
        cv2.imwrite(orig_path, orig_annotated)
        print(f"  Original with boxes → {orig_path}")

        img_results = []
        saved_paths = []

        for label, fn in BLUR_VARIANTS:
            result = benchmark(label, frame, boxes, fn)
            img_results.append(result)

            path = save_blurred(img_key, frame, boxes, fn, label, OUTPUT_DIR)
            saved_paths.append(path)

        all_results[img_key] = img_results

        fastest = min(img_results, key=lambda r: r["avg_ms"])
        slowest = max(img_results, key=lambda r: r["avg_ms"])
        print(f"\n  Fastest : {fastest['label']:<30} {fastest['avg_ms']}ms avg")
        print(f"  Slowest : {slowest['label']:<30} {slowest['avg_ms']}ms avg")
        print(f"  Blurred images saved → {OUTPUT_DIR}/")

    # ── save JSON report ─────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 90}")
    print(f"  JSON report → {report_path}")
    print(f"{'=' * 90}\n")