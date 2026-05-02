import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import psutil
from picamera2 import Picamera2
from flask import Flask, Response

# ============================================================
# CONFIG
# ============================================================
OBJ_MODEL_PATH  = "model/obj_model.tflite"
FACE_MODEL_PATH = "model/face_model.tflite"

OBJ_CONF_THRESH  = 0.5
FACE_CONF_THRESH = 0.5
IOU_THRESH       = 0.45

# Options: "gaussian" | "pixelate" | "median" | "box" | "stack"
BLUR_MODE  = "pixelate"

SHOW_BLUR  = True
SHOW_BOXES = False

COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
TARGET_CLASSES = ["laptop", "tv", "cell phone"]
TARGET_IDS     = [COCO_NAMES.index(c) for c in TARGET_CLASSES]

# ============================================================
# LOAD MODELS
# ============================================================
def load_interpreter(path):
    interp = tflite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

obj_interp  = load_interpreter(OBJ_MODEL_PATH)
face_interp = load_interpreter(FACE_MODEL_PATH)

obj_in   = obj_interp.get_input_details()
obj_out  = obj_interp.get_output_details()
face_in  = face_interp.get_input_details()
face_out = face_interp.get_output_details()

OBJ_H,  OBJ_W  = obj_in[0]["shape"][1],  obj_in[0]["shape"][2]
FACE_H, FACE_W = face_in[0]["shape"][1], face_in[0]["shape"][2]

obj_in_scale,   obj_in_zero   = obj_in[0]["quantization"]
obj_out_scale,  obj_out_zero  = obj_out[0]["quantization"]
face_in_scale,  face_in_zero  = face_in[0]["quantization"]
face_out_scale, face_out_zero = face_out[0]["quantization"]

# ============================================================
# PREPROCESS
# ============================================================
def preprocess(frame, w, h, in_scale, in_zero, dtype):
    img = cv2.resize(frame, (w, h))
    img = img[:, :, ::-1]                      # BGR → RGB
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    if dtype in [np.int8, np.uint8] and in_scale > 0:
        img = np.round(img / in_scale + in_zero).astype(dtype)
    return img

# ============================================================
# INFER (generic)
# ============================================================
def run_infer(interp, inp_details, out_details, frame,
              w, h, in_scale, in_zero, out_scale, out_zero):
    inp = preprocess(frame, w, h, in_scale, in_zero,
                     inp_details[0]["dtype"])
    interp.set_tensor(inp_details[0]["index"], inp)
    interp.invoke()
    output = interp.get_tensor(out_details[0]["index"])
    if out_scale > 0:
        output = (output.astype(np.float32) - out_zero) * out_scale
    return output

# ============================================================
# DECODE — OBJECT  (1, 84, N)
# ============================================================
def decode_obj(output):
    pred        = output[0]
    boxes_xywh  = pred[:4, :].T
    class_scores = pred[4:, :].T
    cls_ids = np.argmax(class_scores, axis=1)
    confs   = class_scores[np.arange(len(cls_ids)), cls_ids]
    mask = (confs >= OBJ_CONF_THRESH) & np.isin(cls_ids, TARGET_IDS)
    if mask.sum() == 0:
        return np.array([])
    boxes_xywh = boxes_xywh[mask]
    confs      = confs[mask]
    cx, cy, w, h = boxes_xywh.T
    x1 = (cx - w/2).clip(0, 1);  y1 = (cy - h/2).clip(0, 1)
    x2 = (cx + w/2).clip(0, 1);  y2 = (cy + h/2).clip(0, 1)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confs.tolist(),
        score_threshold=OBJ_CONF_THRESH, nms_threshold=IOU_THRESH)
    if len(indices) == 0:
        return np.array([])
    return boxes_xyxy[indices.flatten()]

# ============================================================
# DECODE — FACE  (1, 5, N)
# ============================================================
def decode_face(output):
    pred       = output[0]
    boxes_xywh = pred[:4, :].T
    confs      = pred[4,  :]
    mask = confs >= FACE_CONF_THRESH
    if mask.sum() == 0:
        return np.array([])
    boxes_xywh = boxes_xywh[mask]
    confs      = confs[mask]
    cx, cy, w, h = boxes_xywh.T
    x1 = (cx - w/2).clip(0, 1);  y1 = (cy - h/2).clip(0, 1)
    x2 = (cx + w/2).clip(0, 1);  y2 = (cy + h/2).clip(0, 1)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confs.tolist(),
        score_threshold=FACE_CONF_THRESH, nms_threshold=IOU_THRESH)
    if len(indices) == 0:
        return np.array([])
    return boxes_xyxy[indices.flatten()]

# ============================================================
# BLUR IMPLEMENTATIONS
# ============================================================

def _gaussian_blur(roi, strength=51):
    """Classic Gaussian blur — smooth, natural-looking."""
    k = strength | 1   # ensure odd
    return cv2.GaussianBlur(roi, (k, k), 0)

def _pixelate_blur(roi, block_size=20):
    """Pixelation — resize down then back up."""
    h, w = roi.shape[:2]
    bw = max(1, w // block_size)
    bh = max(1, h // block_size)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _median_blur(roi, strength=51):
    """Median blur — preserves edges slightly while anonymising."""
    k = strength | 1
    k = min(k, 99)          # cv2.medianBlur max odd kernel < 100
    # medianBlur requires uint8 or float32 of specific sizes
    roi_u8 = roi.astype(np.uint8) if roi.dtype != np.uint8 else roi
    return cv2.medianBlur(roi_u8, k)

def _box_blur(roi, strength=51):
    """Uniform box (averaging) blur."""
    k = strength | 1
    return cv2.blur(roi, (k, k))

def _stack_blur(roi, strength=51):
    """
    Stack blur — fast approximation to Gaussian using two-pass box blurs.
    Mimics the popular StackBlur algorithm with three sequential box blurs.
    """
    k = max(3, strength | 1)
    # Three-pass box blur approximates a Gaussian (central-limit theorem)
    out = cv2.blur(roi, (k, k))
    out = cv2.blur(out, (k, k))
    out = cv2.blur(out, (k, k))
    return out

BLUR_FNS = {
    "gaussian":  _gaussian_blur,
    "pixelate":  _pixelate_blur,
    "median":    _median_blur,
    "box":       _box_blur,
    "stack":     _stack_blur,
}

def blur_regions(frame, boxes, mode="gaussian"):
    if len(boxes) == 0:
        return frame
    fh, fw = frame.shape[:2]
    blur_fn = BLUR_FNS.get(mode, _gaussian_blur)
    for box in boxes:
        x1 = max(0,  int(box[0] * fw));  y1 = max(0,  int(box[1] * fh))
        x2 = min(fw, int(box[2] * fw));  y2 = min(fh, int(box[3] * fh))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        frame[y1:y2, x1:x2] = blur_fn(roi)
    return frame

# ============================================================
# DRAW BOXES  (debug)
# ============================================================
def draw_boxes(frame, obj_boxes, face_boxes):
    fh, fw = frame.shape[:2]
    for box in obj_boxes:
        x1,y1,x2,y2 = int(box[0]*fw),int(box[1]*fh),int(box[2]*fw),int(box[3]*fh)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
        cv2.putText(frame,"screen",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
    for box in face_boxes:
        x1,y1,x2,y2 = int(box[0]*fw),int(box[1]*fh),int(box[2]*fw),int(box[3]*fh)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)
        cv2.putText(frame,"face",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)
    return frame

# ============================================================
# PERFORMANCE OVERLAY
# ============================================================
# Rolling FPS window
_fps_times = []
_FPS_WINDOW = 30        # frames

def draw_overlay(frame, obj_ms, face_ms, total_ms, fps):
    """
    Draws a semi-transparent HUD panel in the top-left corner showing:
      FPS | Obj latency | Face latency | Total latency | CPU % | RAM %
    """
    h, w = frame.shape[:2]

    cpu  = psutil.cpu_percent(interval=None)
    ram  = psutil.virtual_memory().percent

    lines = [
        (f"FPS     {fps:5.1f}",       (0, 220, 120)),
        (f"Obj    {obj_ms:5.1f} ms",  (80, 200, 255)),
        (f"Face   {face_ms:5.1f} ms", (80, 200, 255)),
        (f"Total  {total_ms:5.1f} ms",(200, 160, 255)),
        (f"CPU    {cpu:5.1f} %",      (255, 200, 80)),
        (f"RAM    {ram:5.1f} %",      (255, 140, 80)),
    ]

    pad_x, pad_y = 10, 10
    line_h       = 22
    box_w        = 180
    box_h        = len(lines) * line_h + pad_y * 2

    # semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad_x, pad_y),
                  (pad_x + box_w, pad_y + box_h),
                  (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # thin accent bar on the left edge of the box
    cv2.rectangle(frame,
                  (pad_x, pad_y),
                  (pad_x + 3, pad_y + box_h),
                  (0, 200, 120), -1)

    for i, (text, color) in enumerate(lines):
        y = pad_y + pad_y + i * line_h
        cv2.putText(frame, text,
                    (pad_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    color, 1, cv2.LINE_AA)

    # blur mode badge bottom-right
    badge_text = f"BLUR: {BLUR_MODE.upper()}"
    (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    bx = w - bw - 16;  by = h - 14
    cv2.rectangle(frame, (bx - 6, by - bh - 4), (bx + bw + 4, by + 4),
                  (20, 20, 20), -1)
    cv2.putText(frame, badge_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)

    return frame

# ============================================================
# CAMERA
# ============================================================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        controls={"FrameRate": 30}
    )
)
picam2.start()

# Warm-up psutil so first cpu_percent() call isn't 0
psutil.cpu_percent(interval=None)

# ============================================================
# FLASK STREAM
# ============================================================
app = Flask(__name__)


def generate():
    global BLUR_MODE
    prev_time = time.perf_counter()
    fps     = 0.0
    ALPHA   = 0.1

    while True:
        t_frame_start = time.perf_counter()

        frame = picam2.capture_array()

        # ── object model ────────────────────────────────────
        t0 = time.perf_counter()
        obj_output = run_infer(
            obj_interp, obj_in, obj_out, frame,
            OBJ_W, OBJ_H,
            obj_in_scale, obj_in_zero,
            obj_out_scale, obj_out_zero
        )
        obj_ms = (time.perf_counter() - t0) * 1000

        # ── face model ───────────────────────────────────────
        t0 = time.perf_counter()
        face_output = run_infer(
            face_interp, face_in, face_out, frame,
            FACE_W, FACE_H,
            face_in_scale, face_in_zero,
            face_out_scale, face_out_zero
        )
        face_ms = (time.perf_counter() - t0) * 1000

        # ── decode ───────────────────────────────────────────
        obj_boxes  = decode_obj(obj_output)
        face_boxes = decode_face(face_output)

        # ── merge boxes ──────────────────────────────────────
        all_boxes = []
        if len(obj_boxes)  > 0: all_boxes.append(obj_boxes)
        if len(face_boxes) > 0: all_boxes.append(face_boxes)
        merged = np.vstack(all_boxes) if all_boxes else np.array([])

        # ── blur + optional boxes ────────────────────────────
        if SHOW_BLUR:
            frame = blur_regions(frame, merged, BLUR_MODE)
        if SHOW_BOXES:
            frame = draw_boxes(frame, obj_boxes, face_boxes)

        total_ms = (time.perf_counter() - t_frame_start) * 1000

        # ── FPS ──────────────────────────────────────────────
        now      = time.perf_counter()
        instant  = 1.0 / (now - prev_time + 1e-9)
        fps      = ALPHA * instant + (1 - ALPHA) * fps
        prev_time = now
        _fps_times.append(now)
        while _fps_times and now - _fps_times[0] > 1.0:
            _fps_times.pop(0)
        fps = len(_fps_times)

        # ── overlay ──────────────────────────────────────────
        frame = draw_overlay(frame, obj_ms, face_ms, total_ms, fps)

        # ── BGR for imencode ─────────────────────────────────
        #frame_bgr = frame[:, :, ::-1]
        _, buffer   = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Open in browser: http://<PI_IP>:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)