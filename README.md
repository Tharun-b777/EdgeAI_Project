# Privacy-Preserving Real-Time Video Stream on Raspberry Pi

A real-time privacy preservation system that runs on a Raspberry Pi using two quantized TFLite models — one for face detection and one for object detection (screens/laptops/phones). Detected regions are anonymised using configurable blur strategies and streamed live over a browser via Flask. The project compares sequential and threaded inference architectures and benchmarks multiple blur algorithms for speed and quality.

---

## Project Structure

```
EdgeAI_Project/
├── src/
│   ├── Threaded.py                  # Main app — threaded inference pipeline
│   ├── Sequential.py                # Main app — sequential inference pipeline
│   ├── yolo_obj_comparison_and_development.ipynb  # Model compression notebook
│   └── Blurring/
│       ├── blur.py                  # Blur benchmarking script
│       ├── blurred_output/          # Benchmark results and blurred images
│       │   ├── benchmark_report.json
│       │   └── *.jpg                # Per-blur-method output images
│       └── *.jpg                    # Test images used for benchmarking
├── model/
│   ├── face_model.tflite            # Quantized INT8 face detection model
│   └── obj_model.tflite             # Quantized INT8 object detection model (YOLO)
├── dataset/
│   ├── augment_dataset.py           # Dataset augmentation script
│   └── data_images.rar              # Raw image dataset (142 images, 5 categories)
└── README.md
```

---

## How It Works

The system runs two TFLite models on each frame from a Pi Camera, later the bounding boxes from both models are combined and blurring is applied on them:

- **Face model** — detects human faces and applies an expanded bounding box to cover side profiles
- **Object model** — detects screens (laptop, TV, cell phone) using YOLO with class filtering

Detected regions are blurred using a configurable strategy (pixelate, box, gaussian, median) and the result is streamed to a browser via Flask MJPEG stream.

Two inference architectures are provided:

| File | Architecture | How it works |
|---|---|---|
| `Sequential.py` | Sequential | Inference + blur + encode happen one after another per frame |
| `Threaded.py` | Threaded | Three threads: capture / inference / encode run concurrently |

The threaded mode decouples camera capture from inference, so the video stream stays smooth even when inference is slow.

---

## Setup and Installation

### Requirements

- Raspberry Pi 4 (or 5) with Pi Camera
- Python 3.9+
- Raspberry Pi OS (64-bit recommended)

### Install dependencies

```bash
pip install tflite-runtime opencv-python-headless flask picamera2 psutil numpy
```

> If `tflite-runtime` is not available for your Python version, use:
> ```bash
> pip install tflite-runtime --extra-index-url https://google-coral.github.io/py-repo/
> ```

### Clone the repository

```bash
git clone <your-repo-url>
cd EdgeAI_Project
```

---

## Running the Application

### Threaded mode

```bash
cd src
python Threaded.py
```

### Sequential mode

```bash
cd src
python Sequential.py
```

Once running, open a browser on any device on the same network and go to:

```
http://<RASPBERRY_PI_IP>:5000
```

---

## Configuration

All tunable parameters are at the top of both `Threaded.py` and `Sequential.py`:

| Parameter | Default | Description |
|---|---|---|
| `BLUR_MODE` | `"pixelate"` | Blur type: `"pixelate"`, `"gaussian"`, `"box"`, `"median"` |
| `PIXEL_BLOCK` | `20` | Block size for pixelation |
| `GAUSSIAN_KERNEL` | `51` | Kernel size for Gaussian blur |
| `BOX_KERNEL` | `101` | Kernel size for box blur |
| `OBJ_CONF_THRESH` | `0.5` | Confidence threshold for object detection |
| `FACE_CONF_THRESH` | `0.5` | Confidence threshold for face detection |
| `FACE_EXPAND_X/Y` | `0.10 / 0.05` | Expand face box outward to cover side profiles |
| `OBJ_PAD_X/Y` | `0.05 / 0.05` | Shrink screen box inward to exclude bezel |
| `SHOW_BOXES` | `False` | Set `True` to show detection bounding boxes for debugging |
| `INFER_SCALE` | `1` | Downscale factor for inference frame (e.g. `0.5` = half resolution) |
| `SKIP_MODE` | `False` | Only infer every `INFER_EVERY` frames (sequential mode only) |

---

## Blur Benchmarking

To benchmark all blur methods on your own images:

```bash
cd src/Blurring
python blur.py
```

Results are saved to `blurred_output/benchmark_report.json` along with a blurred image for each method.

### Benchmark Summary (from `benchmark_report.json`)

| Method | Avg (ms)    |
|---|---|
| Pixelate block=20 | 0.81 |  
| Box k=51 | 1.30 |
| StackBlur k=51 | 1.78 |
| Gaussian k=51 1-pass | 11.81 |
| Median k=51 | 27.23 |
| Gaussian k=151 1-pass | 112.23 | 

**Pixelate and Box blur** are the fastest and most suitable for real-time use on the Pi. **Gaussian k=151** and **Median k=51** are too slow for real-time operation.

---

## Dataset and Augmentation

Unzip the dataset_images file.The dataset contains 142 images across 5 categories:

```
data_images/
├── face/
├── Laptop/
├── Laptop_Face/
├── phones/
└── phonesPerson/
```

To augment the dataset (produces ~568 images, 4 augmentations per image):

```bash
cd dataset
python augment_dataset.py data_images
```

Augmented images are saved to `data_images/augmented/`.

The 4 augmentations applied per image are: original, horizontal flip, brightness decrease, and rotation +15°.

## Image Capture

To capture images directly from the Raspberry Pi camera into a dataset folder:

```bash
python capture.py --folder dataset/data_images/face --count 10
```

Images are saved as `img_001.jpg`, `img_002.jpg`, etc. and auto-numbered so existing images are never overwritten. Use `--preview` to see a live viewfinder before each shot. Point `--folder` at any of the five dataset categories (`face`, `Laptop`, `Laptop_Face`, `phones`, `phonesPerson`) to build up the dataset directly on the device.

---

## Model Compression

The notebook `src/yolo_obj_comparison_and_development.ipynb` documents the process of compressing and quantizing the YOLO object and Face detection models to INT8 TFLite format for deployment on the Pi and validating on the collected images.

---

## AI Tool Usage

Parts of this project including the blur benchmarking script, augmentation script, and threaded pipeline architecture were developed with assistance from Claude (Anthropic). All code has been reviewed and adapted by the team.

---

## External Resources

- [YOLOv8 — Ultralytics](https://github.com/ultralytics/ultralytics)
- [TFLite Runtime](https://www.tensorflow.org/lite/guide/python)
- [Picamera2 Documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
- [OpenCV Blur Methods](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html)