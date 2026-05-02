import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import psutil
import os
import threading
from collections import deque
from picamera2 import Picamera2
from flask import Flask, Response

# ============================================================
# CONFIG — all tunable knobs in one place
# ============================================================

# ── Models ──────────────────────────────────────────────────
# OBJ_MODEL_PATH  = "model/yolov8n_int8.tflite"
# FACE_MODEL_PATH = "model/yolov8n_face_int8.tflite"
OBJ_MODEL_PATH  = "model/obj_model.tflite"
FACE_MODEL_PATH = "model/face_model.tflite"

OBJ_CONF_THRESH  = 0.5
FACE_CONF_THRESH = 0.5
IOU_THRESH       = 0.45



MODE = "threaded"  # "sequential" or "threaded"


INFER_SCALE = 1


SKIP_MODE  = False
INFER_EVERY = 3 

# "pixelate"
# "box"       
# "gaussian" 
BLUR_MODE = "pixelate"

PIXEL_BLOCK = 20

BOX_KERNEL = 101

GAUSSIAN_KERNEL = 51
MEDIAN_KERNEL = 51

# ── Box adjustment ────────────────────────────────────────────
# Screens — tighten inward (YOLO box includes bezel)
OBJ_PAD_X  = 0.05
OBJ_PAD_Y  = 0.05
# Faces — expand outward (side profiles need extra coverage)
FACE_EXPAND_X = 0.10
FACE_EXPAND_Y = 0.05

# ── Display ──────────────────────────────────────────────────
SHOW_BLUR  = True
SHOW_BOXES = False   # set True to verify box alignment

# ── Stats ────────────────────────────────────────────────────
FPS_WINDOW  = 30    # rolling FPS window
PRINT_EVERY = 15    # print terminal stats every N frames

# ============================================================
# COCO / TARGET CLASSES
# ============================================================
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
# SHARED STATE  (threaded mode only)
# ============================================================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw_frame:       np.ndarray | None = None
        self.raw_frame_id:    int   = 0
        self.last_obj_boxes:  np.ndarray = np.array([])
        self.last_face_boxes: np.ndarray = np.array([])
        self.jpeg_bytes:      bytes = b""
        self.fps_now:  float = 0.0
        self.fps_avg:  float = 0.0
        self.lat_obj:  float = 0.0
        self.lat_face: float = 0.0
        self.cpu_pct:  float = 0.0
        self.ram_pct:  float = 0.0
        self.stop_event = threading.Event()

state = SharedState()

# ============================================================
# FPS TRACKER
# ============================================================
frame_times   = deque(maxlen=FPS_WINDOW)
total_frames  = 0
total_elapsed = 0.0

def update_fps(elapsed_s):
    global total_frames, total_elapsed
    frame_times.append(elapsed_s)
    total_frames  += 1
    total_elapsed += elapsed_s
    rolling = len(frame_times) / sum(frame_times) if frame_times else 0.0
    average = total_frames / total_elapsed        if total_elapsed > 0 else 0.0
    return rolling, average

# ============================================================
# PSUTIL HANDLES
# ============================================================
_proc = psutil.Process(os.getpid())

# ============================================================
# PREPROCESS + INFER
# ============================================================
def preprocess(frame, w, h, in_scale, in_zero, dtype):
    img = cv2.resize(frame, (w, h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    if dtype in [np.int8, np.uint8] and in_scale > 0:
        img = np.round(img / in_scale + in_zero).astype(dtype)
    return img

def run_infer(interp, inp_details, out_details, frame,
              w, h, in_scale, in_zero, out_scale, out_zero):
    inp = preprocess(frame, w, h, in_scale, in_zero, inp_details[0]["dtype"])
    interp.set_tensor(inp_details[0]["index"], inp)
    interp.invoke()
    output = interp.get_tensor(out_details[0]["index"])
    if out_scale > 0:
        output = (output.astype(np.float32) - out_zero) * out_scale
    return output

# ============================================================
# DECODE
# ============================================================
def decode_obj(output):
    pred         = output[0]
    boxes_xywh   = pred[:4, :].T
    class_scores = pred[4:, :].T
    cls_ids = np.argmax(class_scores, axis=1)
    confs   = class_scores[np.arange(len(cls_ids)), cls_ids]
    mask = (confs >= OBJ_CONF_THRESH) & np.isin(cls_ids, TARGET_IDS)
    if mask.sum() == 0:
        return np.array([])
    boxes_xywh = boxes_xywh[mask];  confs = confs[mask]
    cx, cy, w, h = boxes_xywh.T
    x1=(cx-w/2).clip(0,1); y1=(cy-h/2).clip(0,1)
    x2=(cx+w/2).clip(0,1); y2=(cy+h/2).clip(0,1)
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confs.tolist(),
                                OBJ_CONF_THRESH, IOU_THRESH)
    return boxes_xyxy[indices.flatten()] if len(indices) else np.array([])

def decode_face(output):
    pred       = output[0]
    boxes_xywh = pred[:4, :].T
    confs      = pred[4,  :]
    mask = confs >= FACE_CONF_THRESH
    if mask.sum() == 0:
        return np.array([])
    boxes_xywh = boxes_xywh[mask];  confs = confs[mask]
    cx, cy, w, h = boxes_xywh.T
    x1=(cx-w/2).clip(0,1); y1=(cy-h/2).clip(0,1)
    x2=(cx+w/2).clip(0,1); y2=(cy+h/2).clip(0,1)
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confs.tolist(),
                                FACE_CONF_THRESH, IOU_THRESH)
    return boxes_xyxy[indices.flatten()] if len(indices) else np.array([])

# ============================================================
# BOX ADJUSTMENT
# ============================================================
def tighten_box(box, px, py):
    x1,y1,x2,y2 = box
    bw=x2-x1; bh=y2-y1
    return [x1+bw*px, y1+bh*py, x2-bw*px, y2-bh*py]

def expand_box(box, ex, ey):
    x1,y1,x2,y2 = box
    bw=x2-x1; bh=y2-y1
    return [max(0.,x1-bw*ex), max(0.,y1-bh*ey),
            min(1.,x2+bw*ex), min(1.,y2+bh*ey)]

# ============================================================
# BLUR IMPLEMENTATIONS
# ============================================================
def _apply_pixelate(roi):
    h, w = roi.shape[:2]
    small = cv2.resize(roi,
                       (max(1, w // PIXEL_BLOCK), max(1, h // PIXEL_BLOCK)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def _apply_median(roi):
    k = MEDIAN_KERNEL if MEDIAN_KERNEL % 2 == 1 else MEDIAN_KERNEL + 1
    k = min(k, 99)
    roi_u8 = roi.astype(np.uint8) if roi.dtype != np.uint8 else roi
    return cv2.medianBlur(roi_u8, k)

def _apply_box(roi):
    return cv2.blur(roi, (BOX_KERNEL, BOX_KERNEL))

def _apply_gaussian(roi):
    k = GAUSSIAN_KERNEL if GAUSSIAN_KERNEL % 2 == 1 else GAUSSIAN_KERNEL + 1
    return cv2.GaussianBlur(roi, (k, k), 0)

# dispatch table — selected by BLUR_MODE
_BLUR_FN = {
    "pixelate": _apply_pixelate,
    "box":      _apply_box,
    "gaussian": _apply_gaussian,
    "median":   _apply_median,    # ← add this
}

def apply_regions(frame, obj_boxes, face_boxes):
    """Apply the selected blur mode to all detected regions."""
    blur_fn = _BLUR_FN.get(BLUR_MODE, _apply_pixelate)
    fh, fw  = frame.shape[:2]

    def apply(box):
        x1=max(0, int(box[0]*fw)); y1=max(0, int(box[1]*fh))
        x2=min(fw,int(box[2]*fw)); y2=min(fh,int(box[3]*fh))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return
        frame[y1:y2, x1:x2] = blur_fn(roi)

    for box in obj_boxes:
        apply(tighten_box(box, OBJ_PAD_X, OBJ_PAD_Y))
    for box in face_boxes:
        apply(expand_box(box, FACE_EXPAND_X, FACE_EXPAND_Y))
    return frame

def draw_boxes(frame, obj_boxes, face_boxes):
    fh, fw = frame.shape[:2]
    for box in obj_boxes:
        tb=tighten_box(box,OBJ_PAD_X,OBJ_PAD_Y)
        x1,y1=int(tb[0]*fw),int(tb[1]*fh)
        x2,y2=int(tb[2]*fw),int(tb[3]*fh)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.putText(frame,"screen",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
    for box in face_boxes:
        eb=expand_box(box,FACE_EXPAND_X,FACE_EXPAND_Y)
        x1,y1=int(eb[0]*fw),int(eb[1]*fh)
        x2,y2=int(eb[2]*fw),int(eb[3]*fh)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),1)
        cv2.putText(frame,"face",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)
    return frame

# ============================================================
# HUD OVERLAY
# ============================================================
def draw_hud(frame, fps_now, fps_avg, lat_obj, lat_face, cpu_pct, ram_pct):
    """Draw 4-line HUD: FPS / blur mode / latency / CPU+RAM."""
    # semi-transparent dark bar behind text for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 88), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    color = (0, 255, 255)
    thick = 1

    blur_label = {
    "pixelate": f"Pixelate  block={PIXEL_BLOCK}",
    "box":      f"Box blur  k={BOX_KERNEL}",
    "gaussian": f"Gaussian  k={GAUSSIAN_KERNEL}",
    "median":   f"Median    k={MEDIAN_KERNEL}",   # ← add this
}.get(BLUR_MODE, BLUR_MODE)

    arch_label = MODE.upper()

    cv2.putText(frame,
        f"FPS  roll:{fps_now:.1f}  avg:{fps_avg:.1f}  [{arch_label}]",
        (8, 18), font, scale, color, thick)
    cv2.putText(frame,
        f"Blur  {blur_label}",
        (8, 36), font, scale, color, thick)
    cv2.putText(frame,
        f"Lat  obj:{lat_obj:.1f}ms  face:{lat_face:.1f}ms  "
        f"total:{lat_obj+lat_face:.1f}ms",
        (8, 54), font, scale, color, thick)
    cv2.putText(frame,
        f"CPU  {cpu_pct:.1f}%   RAM  {ram_pct:.1f}%",
        (8, 72), font, scale, color, thick)

    return frame

# ============================================================
# DOWNSCALE HELPER
# ============================================================
def get_infer_frame(frame):
    if INFER_SCALE != 1.0:
        return cv2.resize(frame, (0, 0),
                          fx=INFER_SCALE, fy=INFER_SCALE,
                          interpolation=cv2.INTER_LINEAR)
    return frame

# ============================================================
# CAMERA
# ============================================================
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
)
picam2.start()

# ============================================================
# FLASK
# ============================================================
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# SEQUENTIAL MODE
# Single loop. Inference runs every INFER_EVERY frames.
# Skip frames reuse last known boxes — fast and simple.
# ─────────────────────────────────────────────────────────────
def generate_sequential():
    last_obj_boxes  = np.array([])
    last_face_boxes = np.array([])
    frame_count     = 0
    print_counter   = 0
    lat_obj = lat_face = 0.0

    while True:
        frame        = picam2.capture_array()
        t_start      = time.perf_counter()
        frame_count += 1

        # ── inference — respects SKIP_MODE ───────────────────
        # SKIP_MODE=True  → only infer every INFER_EVERY frames
        # SKIP_MODE=False → infer every frame
        run_infer_this_frame = (
            not SKIP_MODE or (frame_count % INFER_EVERY == 0)
        )

        if run_infer_this_frame:
            infer_frame = get_infer_frame(frame)

            t0 = time.perf_counter()
            obj_out_raw = run_infer(obj_interp, obj_in, obj_out, infer_frame,
                                    OBJ_W, OBJ_H,
                                    obj_in_scale, obj_in_zero,
                                    obj_out_scale, obj_out_zero)
            lat_obj = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            face_out_raw = run_infer(face_interp, face_in, face_out, infer_frame,
                                     FACE_W, FACE_H,
                                     face_in_scale, face_in_zero,
                                     face_out_scale, face_out_zero)
            lat_face = (time.perf_counter() - t0) * 1000

            last_obj_boxes  = decode_obj(obj_out_raw)
            last_face_boxes = decode_face(face_out_raw)

        if SHOW_BLUR:  frame = apply_regions(frame, last_obj_boxes, last_face_boxes)
        if SHOW_BOXES: frame = draw_boxes(frame, last_obj_boxes, last_face_boxes)

        elapsed          = time.perf_counter() - t_start
        fps_now, fps_avg = update_fps(elapsed)
        cpu_pct          = psutil.cpu_percent(interval=None)
        ram_pct          = psutil.virtual_memory().percent

        frame = draw_hud(frame, fps_now, fps_avg, lat_obj, lat_face,
                         cpu_pct, ram_pct)

        print_counter += 1
        if print_counter >= PRINT_EVERY:
            print_counter = 0
            flag = "INFER" if run_infer_this_frame else "skip "
            cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
            ram_proc  = _proc.memory_info().rss / 1e6
            print(
                f"[{flag}]  FPS roll:{fps_now:5.1f} avg:{fps_avg:5.1f}  |  "
                f"Lat obj:{lat_obj:6.1f}ms face:{lat_face:6.1f}ms "
                f"total:{lat_obj+lat_face:6.1f}ms  |  "
                f"screens:{len(last_obj_boxes)} faces:{len(last_face_boxes)}  |  "
                f"CPU sys:{cpu_pct:5.1f}% cores:{[f'{c:.0f}%' for c in cpu_cores]}  |  "
                f"RAM proc:{ram_proc:.0f}MB sys:{ram_pct:.1f}%"
            )

        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")

# ─────────────────────────────────────────────────────────────
# THREADED MODE — 3 threads
#
# Thread 1 CAPTURE  : camera → raw_frame          (~30fps)
# Thread 2 INFERENCE: models → last_*_boxes       (inference rate)
# Thread 3 ENCODE   : fresh frame + stale boxes   (~30fps) → jpeg
# Flask generate()  : reads jpeg_bytes            (~30fps)
#
# Browser gets smooth video; blur boxes update at inference rate.
# Stale boxes are imperceptible for a blur/pixelate effect.
# ─────────────────────────────────────────────────────────────
def _capture_thread(picam2):
    while not state.stop_event.is_set():
        frame = picam2.capture_array()
        with state.lock:
            state.raw_frame    = frame
            state.raw_frame_id += 1

_infer_frame_times   = deque(maxlen=FPS_WINDOW)
_infer_total_frames  = 0
_infer_total_elapsed = 0.0

def _update_infer_fps(elapsed_s):
    global _infer_total_frames, _infer_total_elapsed
    _infer_frame_times.append(elapsed_s)
    _infer_total_frames  += 1
    _infer_total_elapsed += elapsed_s
    rolling = (len(_infer_frame_times) / sum(_infer_frame_times)
               if _infer_frame_times else 0.0)
    average = (_infer_total_frames / _infer_total_elapsed
               if _infer_total_elapsed > 0 else 0.0)
    return rolling, average

def _inference_thread():
    last_id       = -1
    print_counter = 0

    while not state.stop_event.is_set():
        with state.lock:
            if state.raw_frame is None or state.raw_frame_id == last_id:
                frame = None; frame_id = last_id
            else:
                frame = state.raw_frame.copy(); frame_id = state.raw_frame_id

        if frame is None:
            time.sleep(0.002)
            continue

        last_id = frame_id
        t_start = time.perf_counter()
        infer_frame = get_infer_frame(frame)

        t0 = time.perf_counter()
        obj_out_raw = run_infer(obj_interp, obj_in, obj_out, infer_frame,
                                OBJ_W, OBJ_H,
                                obj_in_scale, obj_in_zero,
                                obj_out_scale, obj_out_zero)
        lat_obj = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        face_out_raw = run_infer(face_interp, face_in, face_out, infer_frame,
                                 FACE_W, FACE_H,
                                 face_in_scale, face_in_zero,
                                 face_out_scale, face_out_zero)
        lat_face = (time.perf_counter() - t0) * 1000

        obj_boxes  = decode_obj(obj_out_raw)
        face_boxes = decode_face(face_out_raw)

        elapsed          = time.perf_counter() - t_start
        fps_now, fps_avg = _update_infer_fps(elapsed)
        cpu_pct          = psutil.cpu_percent(interval=None)
        ram_pct          = psutil.virtual_memory().percent

        with state.lock:
            state.last_obj_boxes  = obj_boxes
            state.last_face_boxes = face_boxes
            state.fps_now  = fps_now
            state.fps_avg  = fps_avg
            state.lat_obj  = lat_obj
            state.lat_face = lat_face
            state.cpu_pct  = cpu_pct
            state.ram_pct  = ram_pct

        print_counter += 1
        if print_counter >= PRINT_EVERY:
            print_counter = 0
            cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
            ram_proc  = _proc.memory_info().rss / 1e6
            print(
                f"[INFER]  FPS roll:{fps_now:5.1f} avg:{fps_avg:5.1f}  |  "
                f"Lat obj:{lat_obj:6.1f}ms face:{lat_face:6.1f}ms "
                f"total:{lat_obj+lat_face:6.1f}ms  |  "
                f"screens:{len(obj_boxes)} faces:{len(face_boxes)}  |  "
                f"CPU sys:{cpu_pct:5.1f}% cores:{[f'{c:.0f}%' for c in cpu_cores]}  |  "
                f"RAM proc:{ram_proc:.0f}MB sys:{ram_pct:.1f}%"
            )

def _encode_thread():
    last_encoded_id = -1

    while not state.stop_event.is_set():
        with state.lock:
            if state.raw_frame is None or state.raw_frame_id == last_encoded_id:
                frame = None; frame_id = last_encoded_id
            else:
                frame      = state.raw_frame.copy()
                frame_id   = state.raw_frame_id
                obj_boxes  = state.last_obj_boxes
                face_boxes = state.last_face_boxes
                fps_now    = state.fps_now
                fps_avg    = state.fps_avg
                lat_obj    = state.lat_obj
                lat_face   = state.lat_face
                cpu_pct    = state.cpu_pct
                ram_pct    = state.ram_pct

        if frame is None:
            time.sleep(0.002)
            continue

        last_encoded_id = frame_id

        if SHOW_BLUR:  frame = apply_regions(frame, obj_boxes, face_boxes)
        if SHOW_BOXES: frame = draw_boxes(frame, obj_boxes, face_boxes)

        frame = draw_hud(frame, fps_now, fps_avg, lat_obj, lat_face,
                         cpu_pct, ram_pct)

        _, buf = cv2.imencode(".jpg", frame)
        with state.lock:
            state.jpeg_bytes = buf.tobytes()

def generate_threaded():
    last_sent: bytes = b""
    while True:
        with state.lock:
            jpeg = state.jpeg_bytes
        if jpeg and jpeg is not last_sent:
            last_sent = jpeg
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + jpeg + b"\r\n")
        else:
            time.sleep(0.005)

# ─────────────────────────────────────────────────────────────
# ROUTE — picks generate function based on MODE
# ─────────────────────────────────────────────────────────────
@app.route("/")
def video_feed():
    gen = generate_sequential if MODE == "sequential" else generate_threaded
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    iw = int(1280 * INFER_SCALE)
    ih = int(720  * INFER_SCALE)

    print("=" * 70)
    print(f"  Mode        : {MODE.upper()}")
    print(f"  Blur        : {BLUR_MODE}  ", end="")
    if BLUR_MODE == "pixelate":
        print(f"block={PIXEL_BLOCK}")
    elif BLUR_MODE == "box":
        print(f"k={BOX_KERNEL}")
    else:
        print(f"k={GAUSSIAN_KERNEL}")
    print(f"  Infer res   : {iw}x{ih}  (scale={INFER_SCALE})")
    if MODE == "sequential":
        if SKIP_MODE:
            print(f"  Skip mode   : ON  (infer every {INFER_EVERY} frames)")
        else:
            print(f"  Skip mode   : OFF (infer every frame)")
    print(f"  Face expand : x={FACE_EXPAND_X}  y={FACE_EXPAND_Y}")
    print(f"  Screen pad  : x={OBJ_PAD_X}  y={OBJ_PAD_Y}")
    print(f"  Open        : http://<PI_IP>:5000")
    print("=" * 70)

    # prime psutil before loop starts
    psutil.cpu_percent(interval=None)
    _proc.cpu_percent(interval=None)

    if MODE == "threaded":
        threads = [
            threading.Thread(target=_capture_thread,   args=(picam2,),
                             daemon=True, name="capture"),
            threading.Thread(target=_inference_thread,
                             daemon=True, name="inference"),
            threading.Thread(target=_encode_thread,
                             daemon=True, name="encode"),
        ]
        for t in threads:
            t.start()
        try:
            app.run(host="0.0.0.0", port=5000, debug=False)
        finally:
            state.stop_event.set()
            for t in threads:
                t.join(timeout=3)
            picam2.stop()
    else:
        try:
            app.run(host="0.0.0.0", port=5000, debug=False)
        finally:
            picam2.stop()