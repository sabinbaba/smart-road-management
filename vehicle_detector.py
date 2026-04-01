"""
Vehicle Detection System using YOLO11
Detects: Normal | Congestion | Accident
Input: Webcam (live camera)
Output: Screen display + console alerts + log file
"""

import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import csv

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
WEBCAM_INDEX         = 0          # Change to 1, 2... if multiple cameras
CONGESTION_THRESHOLD = 6          # Vehicles in frame = congestion
OVERLAP_IOU_THRESHOLD= 0.15       # IoU above this = possible accident
ACCIDENT_MIN_BOXES   = 2          # Minimum overlapping vehicles for accident
LOG_FILE             = "detection_log.csv"
FRAME_SKIP           = 2          # Process every Nth frame (performance)

# YOLO vehicle class IDs (COCO dataset)
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Colors (BGR)
COLOR_NORMAL     = (0, 200, 0)      # Green
COLOR_CONGESTION = (0, 165, 255)    # Orange
COLOR_ACCIDENT   = (0, 0, 220)      # Red
COLOR_BOX        = (255, 220, 50)   # Yellow-ish box
COLOR_TEXT_BG    = (20, 20, 20)

# ─────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────
def init_log(path):
    """Initialize CSV log file with headers."""
    file_exists = os.path.isfile(path)
    f = open(path, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
    return f, writer

def log_event(writer, status, count, max_iou):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# ─────────────────────────────────────────────
#  IoU (Intersection over Union)
# ─────────────────────────────────────────────
def compute_iou(boxA, boxB):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# ─────────────────────────────────────────────
#  SCENE ANALYSIS
# ─────────────────────────────────────────────
def analyze_scene(boxes):
    """
    Given a list of bounding boxes, return:
      - status : 'NORMAL' | 'CONGESTION' | 'ACCIDENT'
      - max_iou: highest pairwise IoU found
    """
    count = len(boxes)
    max_iou = 0.0
    overlap_count = 0

    # Check pairwise IoU for accident detection
    for i in range(count):
        for j in range(i + 1, count):
            iou = compute_iou(boxes[i], boxes[j])
            if iou > max_iou:
                max_iou = iou
            if iou >= OVERLAP_IOU_THRESHOLD:
                overlap_count += 1

    # Decision logic
    if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
        status = "ACCIDENT"
    elif count >= CONGESTION_THRESHOLD:
        status = "CONGESTION"
    else:
        status = "NORMAL"

    return status, max_iou, count

# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def get_status_color(status):
    return {
        "NORMAL":     COLOR_NORMAL,
        "CONGESTION": COLOR_CONGESTION,
        "ACCIDENT":   COLOR_ACCIDENT,
    }.get(status, COLOR_NORMAL)

def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
    """Draw a label with background."""
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

def draw_status_banner(frame, status, count, max_iou, fps):
    """Draw top status banner on frame."""
    h, w = frame.shape[:2]
    color = get_status_color(status)

    # Semi-transparent banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status indicator circle
    cv2.circle(frame, (35, 35), 14, color, -1)
    cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)

    # Status text
    icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
    label = icons.get(status, status)
    cv2.putText(frame, label, (60, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

    # Stats
    stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
    cv2.putText(frame, stats, (w - 380, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom bar: timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

def draw_boxes(frame, detections, status):
    """Draw bounding boxes for each detected vehicle."""
    color = get_status_color(status)
    for (x1, y1, x2, y2), label, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Vehicle Detection System  —  YOLO11")
    print("  Status: NORMAL | CONGESTION | ACCIDENT")
    print("=" * 55)
    print(f"[INFO] Loading YOLO11 model...")

    model = YOLO("yolo11n.pt")   # Downloads automatically on first run
    print("[INFO] Model loaded. Opening webcam...")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    log_f, log_writer = init_log(LOG_FILE)
    print(f"[INFO] Logging to: {LOG_FILE}")
    print("[INFO] Press  Q  to quit.\n")

    frame_idx   = 0
    fps         = 0.0
    t_prev      = datetime.now()
    last_status = "NORMAL"
    last_count  = 0
    last_iou    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        frame_idx += 1

        # ── FPS calculation ──────────────────
        now = datetime.now()
        elapsed = (now - t_prev).total_seconds()
        if elapsed > 0:
            fps = 1.0 / elapsed
        t_prev = now

        # ── Detection (every FRAME_SKIP frames) ─
        if frame_idx % FRAME_SKIP == 0:
            results = model(frame, verbose=False)[0]

            detections = []
            boxes_raw  = []

            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                conf = float(box.conf[0])
                if conf < 0.35:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = VEHICLE_CLASSES[cls_id]
                detections.append(((x1, y1, x2, y2), label, conf))
                boxes_raw.append([x1, y1, x2, y2])

            # ── Scene analysis ───────────────
            status, max_iou, count = analyze_scene(boxes_raw)

            # ── Console alert (on change) ────
            if status != last_status:
                ts = datetime.now().strftime("%H:%M:%S")
                alert_icons = {
                    "NORMAL":     "✅",
                    "CONGESTION": "🚦",
                    "ACCIDENT":   "🚨",
                }
                print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
                      f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")

            # ── Log every 30 frames ──────────
            if frame_idx % 30 == 0:
                log_event(log_writer, status, count, max_iou)
                log_f.flush()

            last_status = status
            last_count  = count
            last_iou    = max_iou

        # ── Draw ────────────────────────────
        if frame_idx % FRAME_SKIP == 0:
            draw_boxes(frame, detections, last_status)

        draw_status_banner(frame, last_status, last_count, last_iou, fps)

        # ── Show ────────────────────────────
        cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit requested by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    log_f.close()
    print(f"[INFO] Log saved to: {LOG_FILE}")
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
