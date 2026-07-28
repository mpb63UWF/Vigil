"""
Live drone detection using YOLOv11x + Arducam 1080P USB camera.

Controls:
  Q / ESC  - quit
  S        - save current frame as PNG
  +/-      - raise/lower confidence threshold
  C        - cycle to next camera index (if wrong camera opens)
"""

import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "best.pt"
CAMERA_INDEX = 0       # change to 1, 2, … if Arducam is not the default camera
CONF_THRESHOLD = 0.35  # initial confidence threshold
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW gives lower latency on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency
    return cap


def draw_overlay(frame, detections, conf_thresh: float, fps: float):
    """Draw bounding boxes, labels, and HUD on the frame."""
    drone_count = 0

    for box in detections.boxes:
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        drone_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Red box, thicker when high confidence
        thickness = 2 if conf < 0.7 else 3
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        label = f"Drone {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # HUD
    hud = [
        f"FPS: {fps:.1f}",
        f"Conf: {conf_thresh:.0%}  (+/- to adjust)",
        f"Drones detected: {drone_count}",
        "Q/ESC: quit   S: save frame   C: next cam",
    ]
    for i, line in enumerate(hud):
        cv2.putText(frame, line, (10, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    return frame, drone_count


def main():
    print(f"Loading model from {MODEL_PATH} …")
    model = YOLO(str(MODEL_PATH))
    print("Model loaded. Opening camera …")

    cam_index = CAMERA_INDEX
    cap = open_camera(cam_index)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {cam_index}. "
              "Try pressing C in the window to cycle cameras, or change CAMERA_INDEX.")
        sys.exit(1)

    print(f"Camera {cam_index} opened. Press Q or ESC to quit.")

    conf = CONF_THRESHOLD
    prev_time = time.perf_counter()
    save_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame — check camera connection.")
            break

        # Run inference (stream=False returns a list with one Results object)
        results = model(frame, verbose=False)[0]

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        frame, _ = draw_overlay(frame, results, conf, fps)

        cv2.imshow("Drone Detection — YOLOv11x", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # Q or ESC
            break
        elif key == ord("s"):
            path = f"capture_{save_count:04d}.png"
            cv2.imwrite(path, frame)
            print(f"Saved {path}")
            save_count += 1
        elif key == ord("+") or key == ord("="):
            conf = min(conf + 0.05, 0.95)
            print(f"Confidence threshold: {conf:.0%}")
        elif key == ord("-"):
            conf = max(conf - 0.05, 0.05)
            print(f"Confidence threshold: {conf:.0%}")
        elif key == ord("c"):
            cam_index += 1
            cap.release()
            cap = open_camera(cam_index)
            if not cap.isOpened():
                cam_index = 0
                cap = open_camera(cam_index)
            print(f"Switched to camera index {cam_index}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
