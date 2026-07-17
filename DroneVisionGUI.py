"""
DroneVisionGUI.py
-----------------
Real-time visual drone detection GUI using YOLOv11x.
Run with:

    python DroneVisionGUI.py

Dependencies:
    pip install ultralytics opencv-python pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import queue
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── Colors (matching DroneDetectorGUI.py) ──────────────────────────────────────
BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#0f3460"
GREEN  = "#00d4aa"
YELLOW = "#f5a623"
RED    = "#e94560"
WHITE  = "#e0e0e0"
GREY   = "#555577"

# ── Defaults ───────────────────────────────────────────────────────────────────
DISPLAY_W     = 640
DISPLAY_H     = 480
_here         = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = r"D:\Drone-Detection-YOLOv11x-main\best.pt"
CONF_DEFAULT  = 0.35
IOU_DEFAULT   = 0.45
INFER_SIZE    = 640            # YOLO input resolution

# COCO classes that can represent drones with a general model.
# airplane=4 is the primary match; kite=33 catches small drones at distance.
# Replace with [0] (or whatever class) if using a custom drone-only model.
DRONE_CLASSES = [0]


def confidence_color(p: float) -> str:
    if p >= 0.75:
        return RED
    if p >= 0.40:
        return YELLOW
    return GREEN


def find_cameras(max_test: int = 8) -> list[int]:
    """Probe camera indices 0..max_test-1 and return those that open."""
    found = []
    for i in range(max_test):
        for backend in (cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(i, backend)
            if cap.isOpened():
                found.append(i)
                cap.release()
                break
    return found or [0]


# ── Camera + YOLO inference thread ─────────────────────────────────────────────
class CameraDetector:
    def __init__(self, result_q: queue.Queue, cam_idx: int,
                 model_path: str, conf_var: tk.DoubleVar, iou_var: tk.DoubleVar):
        self.result_q   = result_q
        self.cam_idx    = cam_idx
        self.model_path = model_path
        self.conf_var   = conf_var
        self.iou_var    = iou_var
        self._stop      = threading.Event()

    def run(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.result_q.put({"error": "ultralytics not installed. Run: pip install ultralytics"})
            return

        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.result_q.put({"error": f"Cannot load model: {e}"})
            return

        cap = None
        for backend in (cv2.CAP_MSMF, cv2.CAP_ANY):
            c = cv2.VideoCapture(self.cam_idx, backend)
            if c.isOpened():
                cap = c
                break
        if cap is None or not cap.isOpened():
            self.result_q.put({"error": f"Cannot open camera {self.cam_idx}"})
            return

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        except Exception:
            pass  # camera doesn't support this resolution — use its default

        prev_time = time.perf_counter()

        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                self.result_q.put({"error": "Camera feed lost"})
                break

            conf = self.conf_var.get()
            iou  = self.iou_var.get()

            results  = model.track(frame, conf=conf, iou=iou,
                                    persist=True, verbose=False,
                                    imgsz=INFER_SIZE, tracker="bytetrack.yaml",
                                    classes=DRONE_CLASSES)
            result   = results[0]
            boxes    = result.boxes

            # Reject boxes that are implausibly large (background false positives)
            # or implausibly small (noise). Valid range: 0.5% – 35% of frame area.
            h_f, w_f = frame.shape[:2]
            frame_area = h_f * w_f
            valid = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                ratio = ((x2 - x1) * (y2 - y1)) / frame_area
                if 0.005 <= ratio <= 0.35:
                    valid.append(i)
            result.boxes = boxes[valid]

            annotated = result.plot()  # BGR with boxes + track IDs drawn

            boxes     = result.boxes
            num_det   = len(boxes)
            max_conf  = float(boxes.conf.max()) if num_det > 0 else 0.0
            track_ids = (boxes.id.int().tolist()
                         if boxes.id is not None else [])

            now       = time.perf_counter()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            payload = {
                "frame":     rgb_frame,
                "num_det":   num_det,
                "max_conf":  max_conf,
                "track_ids": track_ids,
                "fps":       fps,
            }
            try:
                self.result_q.put_nowait(payload)
            except queue.Full:
                pass  # UI is slow — drop frame rather than stalling

        cap.release()

    def stop(self):
        self._stop.set()


# ── Main GUI ────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drone Vision Detector")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.result_q    = queue.Queue(maxsize=4)
        self.detector    = None
        self.det_thread  = None
        self.alert_count = 0
        self._current_photo = None  # keep PhotoImage reference

        self.conf_v  = tk.DoubleVar(value=CONF_DEFAULT)
        self.iou_v   = tk.DoubleVar(value=IOU_DEFAULT)
        self.model_v = tk.StringVar(value=DEFAULT_MODEL)

        self._build_ui()
        self._poll()

    # ── Layout ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        tk.Label(self, text="DRONE VISION DETECTOR",
                 font=("Helvetica", 16, "bold"),
                 bg=BG, fg=WHITE).pack(pady=(16, 4))

        # ── Main area: video canvas + status panel ───────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, **pad)

        # Video canvas
        vid_outer = tk.Frame(main, bg=PANEL, bd=2, relief="flat")
        vid_outer.pack(side="left", padx=(0, 8))

        self.canvas = tk.Canvas(vid_outer, width=DISPLAY_W, height=DISPLAY_H,
                                bg="black", highlightthickness=0)
        self.canvas.pack()
        self._blank_img = ImageTk.PhotoImage(
            Image.new("RGB", (DISPLAY_W, DISPLAY_H), (10, 10, 30))
        )
        self._canvas_img = self.canvas.create_image(0, 0, anchor="nw",
                                                    image=self._blank_img)

        # Status panel
        sp = tk.Frame(main, bg=PANEL, width=200)
        sp.pack(side="left", fill="y")
        sp.pack_propagate(False)

        def _stat_section(label, big=False):
            tk.Label(sp, text=label, font=("Helvetica", 9),
                     bg=PANEL, fg=GREY).pack(pady=(14, 0))
            size = 42 if big else 22
            lbl = tk.Label(sp, text="—", font=("Helvetica", size, "bold"),
                           bg=PANEL, fg=GREEN)
            lbl.pack()
            return lbl

        tk.Label(sp, text="STATUS", font=("Helvetica", 9),
                 bg=PANEL, fg=GREY).pack(pady=(14, 0))
        self.status_label = tk.Label(sp, text="● IDLE",
                                     font=("Helvetica", 16, "bold"),
                                     bg=PANEL, fg=GREY, wraplength=180)
        self.status_label.pack(pady=(4, 0))

        self.count_label = _stat_section("DETECTIONS", big=True)
        self.count_label.config(text="0")

        self.conf_disp = _stat_section("CONFIDENCE")
        self.conf_disp.config(text="—")

        tk.Label(sp, text="ALERTS", font=("Helvetica", 9),
                 bg=PANEL, fg=GREY).pack(pady=(14, 0))
        self.alert_label = tk.Label(sp, text="0",
                                    font=("Helvetica", 22, "bold"),
                                    bg=PANEL, fg=GREY)
        self.alert_label.pack()

        tk.Label(sp, text="FPS", font=("Helvetica", 9),
                 bg=PANEL, fg=GREY).pack(pady=(14, 0))
        self.fps_label = tk.Label(sp, text="—",
                                  font=("Helvetica", 14),
                                  bg=PANEL, fg=GREY)
        self.fps_label.pack()

        tk.Label(sp, text="TRACK IDs", font=("Helvetica", 9),
                 bg=PANEL, fg=GREY).pack(pady=(14, 0))
        self.track_label = tk.Label(sp, text="—",
                                    font=("Helvetica", 11),
                                    bg=PANEL, fg=GREY, wraplength=180)
        self.track_label.pack(pady=(0, 10))

        # ── Controls ─────────────────────────────────────────────────────────
        tk.Frame(self, bg=GREY, height=1).pack(fill="x", padx=12, pady=(6, 0))

        ctrl = tk.Frame(self, bg=BG)
        ctrl.pack(fill="x", **pad)

        # Camera selector
        cam_row = tk.Frame(ctrl, bg=BG)
        cam_row.pack(fill="x", pady=(4, 2))
        tk.Label(cam_row, text="Camera:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")

        cam_indices = find_cameras()
        cam_options = [f"Camera {i}" for i in cam_indices]
        self._cam_map = {f"Camera {i}": i for i in cam_indices}

        self.cam_combo = ttk.Combobox(cam_row, values=cam_options,
                                      width=22, state="readonly")
        # Default to last camera found (Arducam is usually the non-built-in one)
        self.cam_combo.set(cam_options[-1] if cam_options else "Camera 0")
        self.cam_combo.pack(side="left", padx=(4, 0))

        # Model path
        model_row = tk.Frame(ctrl, bg=BG)
        model_row.pack(fill="x", pady=(4, 2))
        tk.Label(model_row, text="Model:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")
        tk.Entry(model_row, textvariable=self.model_v,
                 font=("Helvetica", 9), bg=ACCENT, fg=WHITE,
                 insertbackground=WHITE, width=34,
                 relief="flat").pack(side="left", padx=(4, 4))
        tk.Button(model_row, text="Browse…",
                  font=("Helvetica", 9), bg=PANEL, fg=WHITE, relief="flat",
                  command=self._browse_model).pack(side="left")

        # Confidence threshold
        conf_row = tk.Frame(ctrl, bg=BG)
        conf_row.pack(fill="x", pady=(4, 2))
        tk.Label(conf_row, text="Threshold:", font=("Helvetica", 10),
                 bg=BG, fg=WHITE, width=10, anchor="w").pack(side="left")
        self.thresh_lbl = tk.Label(conf_row, text=f"{CONF_DEFAULT:.2f}",
                                   font=("Helvetica", 10, "bold"),
                                   bg=BG, fg=YELLOW, width=4)
        self.thresh_lbl.pack(side="right")
        tk.Scale(conf_row, variable=self.conf_v,
                 from_=0.05, to=0.95, resolution=0.05,
                 orient="horizontal", bg=BG, fg=WHITE,
                 troughcolor=ACCENT, highlightthickness=0, length=330,
                 command=lambda v: self.thresh_lbl.config(text=f"{float(v):.2f}")
                 ).pack(side="left", expand=True, fill="x")

        # Start / Stop
        btn_row = tk.Frame(ctrl, bg=BG)
        btn_row.pack(pady=(10, 4))
        self.start_btn = tk.Button(btn_row, text="▶  START",
                                   font=("Helvetica", 12, "bold"),
                                   bg=GREEN, fg=BG, relief="flat",
                                   padx=20, pady=6,
                                   command=self.start_detection)
        self.start_btn.pack(side="left", padx=8)
        self.stop_btn = tk.Button(btn_row, text="■  STOP",
                                  font=("Helvetica", 12, "bold"),
                                  bg=RED, fg=WHITE, relief="flat",
                                  padx=20, pady=6, state="disabled",
                                  command=self.stop_detection)
        self.stop_btn.pack(side="left", padx=8)

        self.info_label = tk.Label(self, text="Press START to begin",
                                   font=("Helvetica", 9), bg=BG, fg=GREY)
        self.info_label.pack(pady=(0, 10))

    # ── Actions ─────────────────────────────────────────────────────────────────
    def _browse_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO weights (.pt)",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_v.set(path)

    def start_detection(self):
        cam_name   = self.cam_combo.get()
        cam_idx    = self._cam_map.get(cam_name, 0)
        model_path = self.model_v.get().strip() or DEFAULT_MODEL

        self.alert_count = 0
        self.detector    = CameraDetector(self.result_q, cam_idx, model_path,
                                          self.conf_v, self.iou_v)
        self.det_thread  = threading.Thread(target=self.detector.run, daemon=True)
        self.det_thread.start()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.cam_combo.config(state="disabled")
        self.status_label.config(text="● SCANNING", fg=YELLOW)
        self.info_label.config(text="Initializing model…", fg=YELLOW)

    def stop_detection(self):
        if self.detector:
            self.detector.stop()
            self.detector = None
        self._reset_ui("Stopped.")

    def _reset_ui(self, msg: str = ""):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.cam_combo.config(state="readonly")
        self.status_label.config(text="● IDLE", fg=GREY)
        self.conf_disp.config(text="—", fg=GREEN)
        self.count_label.config(text="0", fg=GREEN)
        self.fps_label.config(text="—", fg=GREY)
        self.alert_label.config(text="0", fg=GREY)
        self.track_label.config(text="—", fg=GREY)
        self.info_label.config(text=msg, fg=GREY)
        self.canvas.itemconfig(self._canvas_img, image=self._blank_img)

    # ── Queue polling ────────────────────────────────────────────────────────────
    def _poll(self):
        # Drain queue; only render the latest frame to avoid lag
        latest = None
        try:
            while True:
                latest = self.result_q.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            if "error" in latest:
                self.info_label.config(text=f"Error: {latest['error'][:70]}", fg=RED)
                self._reset_ui(f"Error: {latest['error'][:60]}")
            else:
                self._render(latest)

        self.after(33, self._poll)  # ~30 fps

    def _render(self, item: dict):
        frame    = item["frame"]      # H×W×3 RGB numpy
        num_det  = item["num_det"]
        max_conf = item["max_conf"]
        fps      = item["fps"]

        # Scale frame to fit canvas while keeping aspect ratio
        h, w   = frame.shape[:2]
        scale  = min(DISPLAY_W / w, DISPLAY_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Center on black canvas
        canvas_arr = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        x0 = (DISPLAY_W - nw) // 2
        y0 = (DISPLAY_H - nh) // 2
        canvas_arr[y0:y0+nh, x0:x0+nw] = resized

        photo = ImageTk.PhotoImage(Image.fromarray(canvas_arr))
        self._current_photo = photo  # prevent garbage collection
        self.canvas.itemconfig(self._canvas_img, image=photo)

        # Status widgets
        self.fps_label.config(text=f"{fps:.1f}", fg=GREY)
        self.info_label.config(text="Running…", fg=GREEN)

        track_ids = item.get("track_ids", [])

        if num_det > 0:
            col = confidence_color(max_conf)
            self.count_label.config(text=str(num_det), fg=col)
            self.conf_disp.config(text=f"{max_conf:.0%}", fg=col)
            self.status_label.config(text="● DRONE DETECTED", fg=RED)
            self.alert_count += 1
            self.alert_label.config(text=str(self.alert_count), fg=RED)
            id_str = ", ".join(str(i) for i in track_ids) if track_ids else "—"
            self.track_label.config(text=id_str, fg=YELLOW)
        else:
            self.count_label.config(text="0", fg=GREEN)
            self.conf_disp.config(text="—", fg=GREEN)
            self.status_label.config(text="● CLEAR", fg=GREEN)
            self.track_label.config(text="—", fg=GREY)


# ── Entry point ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
