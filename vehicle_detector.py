# """
# Vehicle Detection System using YOLO11
# Detects: Normal | Congestion | Accident
# Input: Webcam (live camera)
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX         = 0          # Change to 1, 2... if multiple cameras
# CONGESTION_THRESHOLD = 6          # Vehicles in frame = congestion
# OVERLAP_IOU_THRESHOLD= 0.15       # IoU above this = possible accident
# ACCIDENT_MIN_BOXES   = 2          # Minimum overlapping vehicles for accident
# LOG_FILE             = "detection_log.csv"
# FRAME_SKIP           = 2          # Process every Nth frame (performance)

# # YOLO vehicle class IDs (COCO dataset)
# VEHICLE_CLASSES = {
#     2:  "car",
#     3:  "motorcycle",
#     5:  "bus",
#     7:  "truck",
# }

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)      # Green
# COLOR_CONGESTION = (0, 165, 255)    # Orange
# COLOR_ACCIDENT   = (0, 0, 220)      # Red
# COLOR_BOX        = (255, 220, 50)   # Yellow-ish box
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING SETUP
# # ─────────────────────────────────────────────
# def init_log(path):
#     """Initialize CSV log file with headers."""
#     file_exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     writer = csv.writer(f)
#     if not file_exists:
#         writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
#     return f, writer

# def log_event(writer, status, count, max_iou):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU (Intersection over Union)
# # ─────────────────────────────────────────────
# def compute_iou(boxA, boxB):
#     """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union_area = areaA + areaB - inter_area

#     return inter_area / union_area if union_area > 0 else 0.0

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(boxes):
#     """
#     Given a list of bounding boxes, return:
#       - status : 'NORMAL' | 'CONGESTION' | 'ACCIDENT'
#       - max_iou: highest pairwise IoU found
#     """
#     count = len(boxes)
#     max_iou = 0.0
#     overlap_count = 0

#     # Check pairwise IoU for accident detection
#     for i in range(count):
#         for j in range(i + 1, count):
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > max_iou:
#                 max_iou = iou
#             if iou >= OVERLAP_IOU_THRESHOLD:
#                 overlap_count += 1

#     # Decision logic
#     if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
#         status = "ACCIDENT"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, max_iou, count

# # ─────────────────────────────────────────────
# #  DRAWING HELPERS
# # ─────────────────────────────────────────────
# def get_status_color(status):
#     return {
#         "NORMAL":     COLOR_NORMAL,
#         "CONGESTION": COLOR_CONGESTION,
#         "ACCIDENT":   COLOR_ACCIDENT,
#     }.get(status, COLOR_NORMAL)

# def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
#     """Draw a label with background."""
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness, cv2.LINE_AA)

# def draw_status_banner(frame, status, count, max_iou, fps):
#     """Draw top status banner on frame."""
#     h, w = frame.shape[:2]
#     color = get_status_color(status)

#     # Semi-transparent banner
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     # Status indicator circle
#     cv2.circle(frame, (35, 35), 14, color, -1)
#     cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)

#     # Status text
#     icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
#     label = icons.get(status, status)
#     cv2.putText(frame, label, (60, 45),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

#     # Stats
#     stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w - 380, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     # Bottom bar: timestamp
#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     """Draw bounding boxes for each detected vehicle."""
#     color = get_status_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle Detection System  —  YOLO11")
#     print("  Status: NORMAL | CONGESTION | ACCIDENT")
#     print("=" * 55)
#     print(f"[INFO] Loading YOLO11 model...")

#     model = YOLO("yolo11n.pt")   # Downloads automatically on first run
#     print("[INFO] Model loaded. Opening webcam...")

#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press  Q  to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_iou    = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to read frame from webcam.")
#             break

#         frame_idx += 1

#         # ── FPS calculation ──────────────────
#         now = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         if elapsed > 0:
#             fps = 1.0 / elapsed
#         t_prev = now

#         # ── Detection (every FRAME_SKIP frames) ─
#         if frame_idx % FRAME_SKIP == 0:
#             results = model(frame, verbose=False)[0]

#             detections = []
#             boxes_raw  = []

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = VEHICLE_CLASSES[cls_id]
#                 detections.append(((x1, y1, x2, y2), label, conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Scene analysis ───────────────
#             status, max_iou, count = analyze_scene(boxes_raw)

#             # ── Console alert (on change) ────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 alert_icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                     "ACCIDENT":   "🚨",
#                 }
#                 print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
#                       f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, max_iou)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_iou    = max_iou

#         # ── Draw ────────────────────────────
#         if frame_idx % FRAME_SKIP == 0:
#             draw_boxes(frame, detections, last_status)

#         draw_status_banner(frame, last_status, last_count, last_iou, fps)

#         # ── Show ────────────────────────────
#         cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit requested by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved to: {LOG_FILE}")
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     main()




#==================================end of version 1.0==================================




















# """
# Vehicle Detection System — YOLO11 + Custom Accident Model
# Uses TWO models:
#   1. yolo11n.pt         → detects vehicles (car, truck, bus, motorcycle)
#   2. best.pt (custom)   → classifies accident severity (Moderate / Severe)

# Input : Webcam
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv
# from pathlib import Path

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX          = 0
# CONGESTION_THRESHOLD  = 6
# LOG_FILE              = "detection_log.csv"
# FRAME_SKIP            = 2

# # Model paths
# VEHICLE_MODEL_PATH    = "yolo11n.pt"
# ACCIDENT_MODEL_PATH = r"runs\detect\accident_training\accident_v1\weights\best.pt"

# # YOLO COCO vehicle class IDs
# VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)
# COLOR_CONGESTION = (0, 165, 255)
# COLOR_MODERATE   = (0, 200, 255)
# COLOR_SEVERE     = (0, 0, 220)
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING
# # ─────────────────────────────────────────────
# def init_log(path):
#     exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     w = csv.writer(f)
#     if not exists:
#         w.writerow(["timestamp", "status", "vehicle_count", "accident_class", "confidence"])
#     return f, w

# def log_event(writer, status, count, acc_class, conf):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, acc_class, f"{conf:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU
# # ─────────────────────────────────────────────
# def compute_iou(a, b):
#     xA, yA = max(a[0], b[0]), max(a[1], b[1])
#     xB, yB = min(a[2], b[2]), min(a[3], b[3])
#     inter  = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0.0
#     areaA  = (a[2]-a[0]) * (a[3]-a[1])
#     areaB  = (b[2]-b[0]) * (b[3]-b[1])
#     return inter / (areaA + areaB - inter)

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(vehicle_boxes, accident_results):
#     """
#     Combine vehicle count + custom accident model results.
#     Returns: status, accident_class, confidence
#     """
#     count = len(vehicle_boxes)

#     # Check custom accident model output
#     accident_class = "none"
#     acc_confidence = 0.0

#     if accident_results and len(accident_results[0].boxes) > 0:
#         for box in accident_results[0].boxes:
#             conf     = float(box.conf[0])
#             cls_id   = int(box.cls[0])
#             cls_name = accident_results[0].names[cls_id]
#             if conf > acc_confidence:
#                 acc_confidence = conf
#                 accident_class = cls_name

#     # Decision logic
#     if accident_class in ["Moderate", "Severe"] and acc_confidence > 0.45:
#         status = f"ACCIDENT ({accident_class.upper()})"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, accident_class, acc_confidence, count

# # ─────────────────────────────────────────────
# #  DRAWING
# # ─────────────────────────────────────────────
# def get_color(status):
#     if "SEVERE" in status:    return COLOR_SEVERE
#     if "MODERATE" in status:  return COLOR_MODERATE
#     if "CONGESTION" in status:return COLOR_CONGESTION
#     return COLOR_NORMAL

# def draw_label(img, text, pos, color):
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

# def draw_banner(frame, status, count, acc_class, conf, fps):
#     h, w = frame.shape[:2]
#     color = get_color(status)

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     cv2.circle(frame, (35, 37), 15, color, -1)
#     cv2.circle(frame, (35, 37), 15, (255,255,255), 1)

#     icons = {
#         "NORMAL":     "✓ NORMAL",
#         "CONGESTION": "⚠ CONGESTION",
#     }
#     label = icons.get(status, f"✖ {status}")
#     cv2.putText(frame, label, (62, 50),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.05, color, 2, cv2.LINE_AA)

#     stats = f"Vehicles: {count}   Conf: {conf:.0%}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w-390, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     color = get_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle + Accident Detection — YOLO11")
#     print("  Classes: Normal | Congestion | Moderate | Severe")
#     print("=" * 55)

#     # Load vehicle detection model
#     print("[INFO] Loading vehicle detection model...")
#     vehicle_model = YOLO(VEHICLE_MODEL_PATH)

#     # Load custom accident model
#     accident_model = None
#     if Path(ACCIDENT_MODEL_PATH).exists():
#         print("[INFO] Loading custom accident model...")
#         accident_model = YOLO(ACCIDENT_MODEL_PATH)
#         print("[INFO] ✅ Custom accident model loaded!")
#     else:
#         print(f"[WARN] ⚠️  Custom model not found at: {ACCIDENT_MODEL_PATH}")
#         print("[WARN]    Train first using train_accident_model.py")
#         print("[WARN]    Running with vehicle detection only...\n")

#     # Open webcam
#     print("[INFO] Opening webcam...")
#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print("[ERROR] Cannot open webcam!")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press Q to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_conf   = 0.0
#     last_class  = "none"
#     detections  = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_idx += 1
#         now     = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         fps     = 1.0 / elapsed if elapsed > 0 else fps
#         t_prev  = now

#         if frame_idx % FRAME_SKIP == 0:

#             # ── Vehicle detection ────────────
#             v_results  = vehicle_model(frame, verbose=False)[0]
#             detections = []
#             boxes_raw  = []

#             for box in v_results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append(((x1, y1, x2, y2), VEHICLE_CLASSES[cls_id], conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Accident model ───────────────
#             acc_results = None
#             if accident_model:
#                 acc_results = accident_model(frame, verbose=False)

#             # ── Scene analysis ───────────────
#             status, acc_class, acc_conf, count = analyze_scene(boxes_raw, acc_results)

#             # ── Console alert ────────────────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                 }
#                 icon = icons.get(status, "🚨")
#                 print(f"[{ts}] {icon} STATUS → {status} "
#                       f"| Vehicles: {count} | Confidence: {acc_conf:.0%}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, acc_class, acc_conf)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_conf   = acc_conf
#             last_class  = acc_class

#         # ── Draw ────────────────────────────
#         draw_boxes(frame, detections, last_status)
#         draw_banner(frame, last_status, last_count, last_class, last_conf, fps)

#         cv2.imshow("Vehicle + Accident Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved: {LOG_FILE}")

# if __name__ == "__main__":
#     main()





# """
# Vehicle Detection System using YOLO11
# Detects: Normal | Congestion | Accident
# Input: Webcam (live camera)
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX         = 0          # Change to 1, 2... if multiple cameras
# CONGESTION_THRESHOLD = 6          # Vehicles in frame = congestion
# OVERLAP_IOU_THRESHOLD= 0.15       # IoU above this = possible accident
# ACCIDENT_MIN_BOXES   = 2          # Minimum overlapping vehicles for accident
# LOG_FILE             = "detection_log.csv"
# FRAME_SKIP           = 2          # Process every Nth frame (performance)

# # YOLO vehicle class IDs (COCO dataset)
# VEHICLE_CLASSES = {
#     2:  "car",
#     3:  "motorcycle",
#     5:  "bus",
#     7:  "truck",
# }

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)      # Green
# COLOR_CONGESTION = (0, 165, 255)    # Orange
# COLOR_ACCIDENT   = (0, 0, 220)      # Red
# COLOR_BOX        = (255, 220, 50)   # Yellow-ish box
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING SETUP
# # ─────────────────────────────────────────────
# def init_log(path):
#     """Initialize CSV log file with headers."""
#     file_exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     writer = csv.writer(f)
#     if not file_exists:
#         writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
#     return f, writer

# def log_event(writer, status, count, max_iou):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU (Intersection over Union)
# # ─────────────────────────────────────────────
# def compute_iou(boxA, boxB):
#     """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union_area = areaA + areaB - inter_area

#     return inter_area / union_area if union_area > 0 else 0.0

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(boxes):
#     """
#     Given a list of bounding boxes, return:
#       - status : 'NORMAL' | 'CONGESTION' | 'ACCIDENT'
#       - max_iou: highest pairwise IoU found
#     """
#     count = len(boxes)
#     max_iou = 0.0
#     overlap_count = 0

#     # Check pairwise IoU for accident detection
#     for i in range(count):
#         for j in range(i + 1, count):
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > max_iou:
#                 max_iou = iou
#             if iou >= OVERLAP_IOU_THRESHOLD:
#                 overlap_count += 1

#     # Decision logic
#     if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
#         status = "ACCIDENT"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, max_iou, count

# # ─────────────────────────────────────────────
# #  DRAWING HELPERS
# # ─────────────────────────────────────────────
# def get_status_color(status):
#     return {
#         "NORMAL":     COLOR_NORMAL,
#         "CONGESTION": COLOR_CONGESTION,
#         "ACCIDENT":   COLOR_ACCIDENT,
#     }.get(status, COLOR_NORMAL)

# def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
#     """Draw a label with background."""
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness, cv2.LINE_AA)

# def draw_status_banner(frame, status, count, max_iou, fps):
#     """Draw top status banner on frame."""
#     h, w = frame.shape[:2]
#     color = get_status_color(status)

#     # Semi-transparent banner
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     # Status indicator circle
#     cv2.circle(frame, (35, 35), 14, color, -1)
#     cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)

#     # Status text
#     icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
#     label = icons.get(status, status)
#     cv2.putText(frame, label, (60, 45),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

#     # Stats
#     stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w - 380, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     # Bottom bar: timestamp
#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     """Draw bounding boxes for each detected vehicle."""
#     color = get_status_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle Detection System  —  YOLO11")
#     print("  Status: NORMAL | CONGESTION | ACCIDENT")
#     print("=" * 55)
#     print(f"[INFO] Loading YOLO11 model...")

#     model = YOLO("yolo11n.pt")   # Downloads automatically on first run
#     print("[INFO] Model loaded. Opening webcam...")

#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press  Q  to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_iou    = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to read frame from webcam.")
#             break

#         frame_idx += 1

#         # ── FPS calculation ──────────────────
#         now = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         if elapsed > 0:
#             fps = 1.0 / elapsed
#         t_prev = now

#         # ── Detection (every FRAME_SKIP frames) ─
#         if frame_idx % FRAME_SKIP == 0:
#             results = model(frame, verbose=False)[0]

#             detections = []
#             boxes_raw  = []

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = VEHICLE_CLASSES[cls_id]
#                 detections.append(((x1, y1, x2, y2), label, conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Scene analysis ───────────────
#             status, max_iou, count = analyze_scene(boxes_raw)

#             # ── Console alert (on change) ────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 alert_icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                     "ACCIDENT":   "🚨",
#                 }
#                 print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
#                       f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, max_iou)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_iou    = max_iou

#         # ── Draw ────────────────────────────
#         if frame_idx % FRAME_SKIP == 0:
#             draw_boxes(frame, detections, last_status)

#         draw_status_banner(frame, last_status, last_count, last_iou, fps)

#         # ── Show ────────────────────────────
#         cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit requested by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved to: {LOG_FILE}")
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     main()




#==================================end of version 1.0==================================


# """
# Vehicle Detection System — YOLO11 + Custom Accident Model
# Uses TWO models:
#   1. yolo11n.pt         → detects vehicles (car, truck, bus, motorcycle)
#   2. best.pt (custom)   → classifies accident severity (Moderate / Severe)

# Input : Webcam
# Output: Screen display + console alerts + log file

# Fix v1.1: Accident detection now requires at least 1 vehicle in frame.
#           No vehicles = forced NORMAL, regardless of accident model output.
# """







# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv
# from pathlib import Path

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX          = 0
# CONGESTION_THRESHOLD  = 6
# LOG_FILE              = "detection_log.csv"
# FRAME_SKIP            = 2

# # Model paths
# VEHICLE_MODEL_PATH  = "yolo11n.pt"
# ACCIDENT_MODEL_PATH = r"runs\detect\accident_training\accident_v1\weights\best.pt"

# # YOLO COCO vehicle class IDs
# VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)
# COLOR_CONGESTION = (0, 165, 255)
# COLOR_MODERATE   = (0, 200, 255)
# COLOR_SEVERE     = (0, 0, 220)
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING
# # ─────────────────────────────────────────────
# def init_log(path):
#     exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     w = csv.writer(f)
#     if not exists:
#         w.writerow(["timestamp", "status", "vehicle_count", "accident_class", "confidence"])
#     return f, w

# def log_event(writer, status, count, acc_class, conf):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, acc_class, f"{conf:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU
# # ─────────────────────────────────────────────
# def compute_iou(a, b):
#     xA, yA = max(a[0], b[0]), max(a[1], b[1])
#     xB, yB = min(a[2], b[2]), min(a[3], b[3])
#     inter  = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0.0
#     areaA = (a[2]-a[0]) * (a[3]-a[1])
#     areaB = (b[2]-b[0]) * (b[3]-b[1])
#     return inter / (areaA + areaB - inter)

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS  ← BUG FIXED HERE
# # ─────────────────────────────────────────────
# def analyze_scene(vehicle_boxes, accident_results):
#     """
#     Combine vehicle count + custom accident model results.
#     Returns: status, accident_class, confidence, count

#     FIX: If no vehicles are detected, status is forced to NORMAL
#          regardless of what the accident model outputs.
#     """
#     count = len(vehicle_boxes)

#     # ── Gate: no vehicles = no accident possible ──
#     if count == 0:
#         return "NORMAL", "none", 0.0, 0

#     # Check custom accident model output
#     accident_class = "none"
#     acc_confidence = 0.0

#     if accident_results and len(accident_results[0].boxes) > 0:
#         for box in accident_results[0].boxes:
#             conf     = float(box.conf[0])
#             cls_id   = int(box.cls[0])
#             cls_name = accident_results[0].names[cls_id]
#             if conf > acc_confidence:
#                 acc_confidence = conf
#                 accident_class = cls_name

#     # Decision logic
#     if accident_class in ["Moderate", "Severe"] and acc_confidence > 0.45:
#         status = f"ACCIDENT ({accident_class.upper()})"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, accident_class, acc_confidence, count

# # ─────────────────────────────────────────────
# #  DRAWING
# # ─────────────────────────────────────────────
# def get_color(status):
#     if "SEVERE" in status:     return COLOR_SEVERE
#     if "MODERATE" in status:   return COLOR_MODERATE
#     if "CONGESTION" in status: return COLOR_CONGESTION
#     return COLOR_NORMAL

# def draw_label(img, text, pos, color):
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

# def draw_banner(frame, status, count, acc_class, conf, fps):
#     h, w = frame.shape[:2]
#     color = get_color(status)

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     cv2.circle(frame, (35, 37), 15, color, -1)
#     cv2.circle(frame, (35, 37), 15, (255, 255, 255), 1)

#     icons = {
#         "NORMAL":     "✓ NORMAL",
#         "CONGESTION": "⚠ CONGESTION",
#     }
#     label = icons.get(status, f"✖ {status}")
#     cv2.putText(frame, label, (62, 50),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.05, color, 2, cv2.LINE_AA)

#     stats = f"Vehicles: {count}   Conf: {conf:.0%}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w-390, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     color = get_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle + Accident Detection — YOLO11")
#     print("  Classes: Normal | Congestion | Moderate | Severe")
#     print("=" * 55)

#     # Load vehicle detection model
#     print("[INFO] Loading vehicle detection model...")
#     vehicle_model = YOLO(VEHICLE_MODEL_PATH)

#     # Load custom accident model
#     accident_model = None
#     if Path(ACCIDENT_MODEL_PATH).exists():
#         print("[INFO] Loading custom accident model...")
#         accident_model = YOLO(ACCIDENT_MODEL_PATH)
#         print("[INFO] ✅ Custom accident model loaded!")
#     else:
#         print(f"[WARN] ⚠️  Custom model not found at: {ACCIDENT_MODEL_PATH}")
#         print("[WARN]    Train first using train_accident_model.py")
#         print("[WARN]    Running with vehicle detection only...\n")

#     # Open webcam
#     print("[INFO] Opening webcam...")
#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print("[ERROR] Cannot open webcam!")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press Q to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_conf   = 0.0
#     last_class  = "none"
#     detections  = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_idx += 1
#         now     = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         fps     = 1.0 / elapsed if elapsed > 0 else fps
#         t_prev  = now

#         if frame_idx % FRAME_SKIP == 0:

#             # ── Vehicle detection ────────────
#             v_results  = vehicle_model(frame, verbose=False)[0]
#             detections = []
#             boxes_raw  = []

#             for box in v_results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append(((x1, y1, x2, y2), VEHICLE_CLASSES[cls_id], conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Accident model ───────────────
#             acc_results = None
#             if accident_model:
#                 acc_results = accident_model(frame, verbose=False)

#             # ── Scene analysis ───────────────
#             status, acc_class, acc_conf, count = analyze_scene(boxes_raw, acc_results)

#             # ── Console alert ────────────────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                 }
#                 icon = icons.get(status, "🚨")
#                 print(f"[{ts}] {icon} STATUS → {status} "
#                       f"| Vehicles: {count} | Confidence: {acc_conf:.0%}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, acc_class, acc_conf)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_conf   = acc_conf
#             last_class  = acc_class

#         # ── Draw ────────────────────────────
#         draw_boxes(frame, detections, last_status)
#         draw_banner(frame, last_status, last_count, last_class, last_conf, fps)

#         cv2.imshow("Vehicle + Accident Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved: {LOG_FILE}")

# if __name__ == "__main__":
#     main()











# ==================================end of version 2==================================











# """
# Vehicle Detection System using YOLO11
# Detects: Normal | Congestion | Accident
# Input: Webcam (live camera)
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX         = 0          # Change to 1, 2... if multiple cameras
# CONGESTION_THRESHOLD = 6          # Vehicles in frame = congestion
# OVERLAP_IOU_THRESHOLD= 0.15       # IoU above this = possible accident
# ACCIDENT_MIN_BOXES   = 2          # Minimum overlapping vehicles for accident
# LOG_FILE             = "detection_log.csv"
# FRAME_SKIP           = 2          # Process every Nth frame (performance)

# # YOLO vehicle class IDs (COCO dataset)
# VEHICLE_CLASSES = {
#     2:  "car",
#     3:  "motorcycle",
#     5:  "bus",
#     7:  "truck",
# }

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)      # Green
# COLOR_CONGESTION = (0, 165, 255)    # Orange
# COLOR_ACCIDENT   = (0, 0, 220)      # Red
# COLOR_BOX        = (255, 220, 50)   # Yellow-ish box
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING SETUP
# # ─────────────────────────────────────────────
# def init_log(path):
#     """Initialize CSV log file with headers."""
#     file_exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     writer = csv.writer(f)
#     if not file_exists:
#         writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
#     return f, writer

# def log_event(writer, status, count, max_iou):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU (Intersection over Union)
# # ─────────────────────────────────────────────
# def compute_iou(boxA, boxB):
#     """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union_area = areaA + areaB - inter_area

#     return inter_area / union_area if union_area > 0 else 0.0

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(boxes):
#     """
#     Given a list of bounding boxes, return:
#       - status : 'NORMAL' | 'CONGESTION' | 'ACCIDENT'
#       - max_iou: highest pairwise IoU found
#     """
#     count = len(boxes)
#     max_iou = 0.0
#     overlap_count = 0

#     # Check pairwise IoU for accident detection
#     for i in range(count):
#         for j in range(i + 1, count):
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > max_iou:
#                 max_iou = iou
#             if iou >= OVERLAP_IOU_THRESHOLD:
#                 overlap_count += 1

#     # Decision logic
#     if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
#         status = "ACCIDENT"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, max_iou, count

# # ─────────────────────────────────────────────
# #  DRAWING HELPERS
# # ─────────────────────────────────────────────
# def get_status_color(status):
#     return {
#         "NORMAL":     COLOR_NORMAL,
#         "CONGESTION": COLOR_CONGESTION,
#         "ACCIDENT":   COLOR_ACCIDENT,
#     }.get(status, COLOR_NORMAL)

# def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
#     """Draw a label with background."""
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness, cv2.LINE_AA)

# def draw_status_banner(frame, status, count, max_iou, fps):
#     """Draw top status banner on frame."""
#     h, w = frame.shape[:2]
#     color = get_status_color(status)

#     # Semi-transparent banner
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     # Status indicator circle
#     cv2.circle(frame, (35, 35), 14, color, -1)
#     cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)

#     # Status text
#     icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
#     label = icons.get(status, status)
#     cv2.putText(frame, label, (60, 45),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

#     # Stats
#     stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w - 380, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     # Bottom bar: timestamp
#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     """Draw bounding boxes for each detected vehicle."""
#     color = get_status_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle Detection System  —  YOLO11")
#     print("  Status: NORMAL | CONGESTION | ACCIDENT")
#     print("=" * 55)
#     print(f"[INFO] Loading YOLO11 model...")

#     model = YOLO("yolo11n.pt")   # Downloads automatically on first run
#     print("[INFO] Model loaded. Opening webcam...")

#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press  Q  to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_iou    = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to read frame from webcam.")
#             break

#         frame_idx += 1

#         # ── FPS calculation ──────────────────
#         now = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         if elapsed > 0:
#             fps = 1.0 / elapsed
#         t_prev = now

#         # ── Detection (every FRAME_SKIP frames) ─
#         if frame_idx % FRAME_SKIP == 0:
#             results = model(frame, verbose=False)[0]

#             detections = []
#             boxes_raw  = []

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = VEHICLE_CLASSES[cls_id]
#                 detections.append(((x1, y1, x2, y2), label, conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Scene analysis ───────────────
#             status, max_iou, count = analyze_scene(boxes_raw)

#             # ── Console alert (on change) ────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 alert_icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                     "ACCIDENT":   "🚨",
#                 }
#                 print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
#                       f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, max_iou)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_iou    = max_iou

#         # ── Draw ────────────────────────────
#         if frame_idx % FRAME_SKIP == 0:
#             draw_boxes(frame, detections, last_status)

#         draw_status_banner(frame, last_status, last_count, last_iou, fps)

#         # ── Show ────────────────────────────
#         cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit requested by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved to: {LOG_FILE}")
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     main()




#==================================end of version 1.1==================================


# """
# Vehicle Detection System — YOLO11 + Custom Accident Model
# Uses TWO models:
#   1. yolo11n.pt         → detects vehicles (car, truck, bus, motorcycle)
#   2. best.pt (custom)   → classifies accident severity (Moderate / Severe)

# Input : Webcam
# Output: Screen display + console alerts + log file

# Fix v1.1: Accident detection now requires at least 1 vehicle in frame.
#           No vehicles = forced NORMAL, regardless of accident model output.
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv
# from pathlib import Path

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX          = 0
# CONGESTION_THRESHOLD  = 6
# LOG_FILE              = "detection_log.csv"
# FRAME_SKIP            = 2

# # Model paths
# VEHICLE_MODEL_PATH  = "yolo11n.pt"
# ACCIDENT_MODEL_PATH = r"runs\detect\accident_training\accident_v1\weights\best.pt"

# # YOLO COCO vehicle class IDs
# VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)
# COLOR_CONGESTION = (0, 165, 255)
# COLOR_MODERATE   = (0, 200, 255)
# COLOR_SEVERE     = (0, 0, 220)
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING
# # ─────────────────────────────────────────────
# def init_log(path):
#     exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     w = csv.writer(f)
#     if not exists:
#         w.writerow(["timestamp", "status", "vehicle_count", "accident_class", "confidence"])
#     return f, w

# def log_event(writer, status, count, acc_class, conf):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, acc_class, f"{conf:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU
# # ─────────────────────────────────────────────
# def compute_iou(a, b):
#     xA, yA = max(a[0], b[0]), max(a[1], b[1])
#     xB, yB = min(a[2], b[2]), min(a[3], b[3])
#     inter  = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0.0
#     areaA = (a[2]-a[0]) * (a[3]-a[1])
#     areaB = (b[2]-b[0]) * (b[3]-b[1])
#     return inter / (areaA + areaB - inter)

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS  ← BUG FIXED HERE
# # ─────────────────────────────────────────────
# def analyze_scene(vehicle_boxes, accident_results):
#     """
#     Combine vehicle count + custom accident model results.
#     Returns: status, accident_class, confidence, count

#     FIX: If no vehicles are detected, status is forced to NORMAL
#          regardless of what the accident model outputs.
#     """
#     count = len(vehicle_boxes)

#     # ── Gate: no vehicles = no accident possible ──
#     if count == 0:
#         return "NORMAL", "none", 0.0, 0

#     # Check custom accident model output
#     accident_class = "none"
#     acc_confidence = 0.0

#     if accident_results and len(accident_results[0].boxes) > 0:
#         for box in accident_results[0].boxes:
#             conf     = float(box.conf[0])
#             cls_id   = int(box.cls[0])
#             cls_name = accident_results[0].names[cls_id]
#             if conf > acc_confidence:
#                 acc_confidence = conf
#                 accident_class = cls_name

#     # Decision logic
#     if accident_class in ["Moderate", "Severe"] and acc_confidence > 0.45:
#         status = f"ACCIDENT ({accident_class.upper()})"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, accident_class, acc_confidence, count

# # ─────────────────────────────────────────────
# #  DRAWING
# # ─────────────────────────────────────────────
# def get_color(status):
#     if "SEVERE" in status:     return COLOR_SEVERE
#     if "MODERATE" in status:   return COLOR_MODERATE
#     if "CONGESTION" in status: return COLOR_CONGESTION
#     return COLOR_NORMAL

# def draw_label(img, text, pos, color):
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

# def draw_banner(frame, status, count, acc_class, conf, fps):
#     h, w = frame.shape[:2]
#     color = get_color(status)

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     cv2.circle(frame, (35, 37), 15, color, -1)
#     cv2.circle(frame, (35, 37), 15, (255, 255, 255), 1)

#     icons = {
#         "NORMAL":     "✓ NORMAL",
#         "CONGESTION": "⚠ CONGESTION",
#     }
#     label = icons.get(status, f"✖ {status}")
#     cv2.putText(frame, label, (62, 50),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.05, color, 2, cv2.LINE_AA)

#     stats = f"Vehicles: {count}   Conf: {conf:.0%}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w-390, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     color = get_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle + Accident Detection — YOLO11")
#     print("  Classes: Normal | Congestion | Moderate | Severe")
#     print("=" * 55)

#     # Load vehicle detection model
#     print("[INFO] Loading vehicle detection model...")
#     vehicle_model = YOLO(VEHICLE_MODEL_PATH)

#     # Load custom accident model
#     accident_model = None
#     if Path(ACCIDENT_MODEL_PATH).exists():
#         print("[INFO] Loading custom accident model...")
#         accident_model = YOLO(ACCIDENT_MODEL_PATH)
#         print("[INFO] ✅ Custom accident model loaded!")
#     else:
#         print(f"[WARN] ⚠️  Custom model not found at: {ACCIDENT_MODEL_PATH}")
#         print("[WARN]    Train first using train_accident_model.py")
#         print("[WARN]    Running with vehicle detection only...\n")

#     # Open webcam
#     print("[INFO] Opening webcam...")
#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print("[ERROR] Cannot open webcam!")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press Q to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_conf   = 0.0
#     last_class  = "none"
#     detections  = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_idx += 1
#         now     = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         fps     = 1.0 / elapsed if elapsed > 0 else fps
#         t_prev  = now

#         if frame_idx % FRAME_SKIP == 0:

#             # ── Vehicle detection ────────────
#             v_results  = vehicle_model(frame, verbose=False)[0]
#             detections = []
#             boxes_raw  = []

#             for box in v_results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append(((x1, y1, x2, y2), VEHICLE_CLASSES[cls_id], conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Accident model ───────────────
#             acc_results = None
#             if accident_model:
#                 acc_results = accident_model(frame, verbose=False)

#             # ── Scene analysis ───────────────
#             status, acc_class, acc_conf, count = analyze_scene(boxes_raw, acc_results)

#             # ── Console alert ────────────────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                 }
#                 icon = icons.get(status, "🚨")
#                 print(f"[{ts}] {icon} STATUS → {status} "
#                       f"| Vehicles: {count} | Confidence: {acc_conf:.0%}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, acc_class, acc_conf)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_conf   = acc_conf
#             last_class  = acc_class

#         # ── Draw ────────────────────────────
#         draw_boxes(frame, detections, last_status)
#         draw_banner(frame, last_status, last_count, last_class, last_conf, fps)

#         cv2.imshow("Vehicle + Accident Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved: {LOG_FILE}")

# if __name__ == "__main__":
#     main()






#==================================end of version 1.2==================================






# """
# Vehicle Detection System using YOLO11
# Detects: Normal | Congestion | Accident
# Input: Webcam (live camera)
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX         = 0          # Change to 1, 2... if multiple cameras
# CONGESTION_THRESHOLD = 6          # Vehicles in frame = congestion
# OVERLAP_IOU_THRESHOLD= 0.15       # IoU above this = possible accident
# ACCIDENT_MIN_BOXES   = 2          # Minimum overlapping vehicles for accident
# LOG_FILE             = "detection_log.csv"
# FRAME_SKIP           = 2          # Process every Nth frame (performance)

# # YOLO vehicle class IDs (COCO dataset)
# VEHICLE_CLASSES = {
#     2:  "car",
#     3:  "motorcycle",
#     5:  "bus",
#     7:  "truck",
# }

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)      # Green
# COLOR_CONGESTION = (0, 165, 255)    # Orange
# COLOR_ACCIDENT   = (0, 0, 220)      # Red
# COLOR_BOX        = (255, 220, 50)   # Yellow-ish box
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING SETUP
# # ─────────────────────────────────────────────
# def init_log(path):
#     """Initialize CSV log file with headers."""
#     file_exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     writer = csv.writer(f)
#     if not file_exists:
#         writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
#     return f, writer

# def log_event(writer, status, count, max_iou):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU (Intersection over Union)
# # ─────────────────────────────────────────────
# def compute_iou(boxA, boxB):
#     """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union_area = areaA + areaB - inter_area

#     return inter_area / union_area if union_area > 0 else 0.0

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(boxes):
#     """
#     Given a list of bounding boxes, return:
#       - status : 'NORMAL' | 'CONGESTION' | 'ACCIDENT'
#       - max_iou: highest pairwise IoU found
#     """
#     count = len(boxes)
#     max_iou = 0.0
#     overlap_count = 0

#     # Check pairwise IoU for accident detection
#     for i in range(count):
#         for j in range(i + 1, count):
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > max_iou:
#                 max_iou = iou
#             if iou >= OVERLAP_IOU_THRESHOLD:
#                 overlap_count += 1

#     # Decision logic
#     if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
#         status = "ACCIDENT"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, max_iou, count

# # ─────────────────────────────────────────────
# #  DRAWING HELPERS
# # ─────────────────────────────────────────────
# def get_status_color(status):
#     return {
#         "NORMAL":     COLOR_NORMAL,
#         "CONGESTION": COLOR_CONGESTION,
#         "ACCIDENT":   COLOR_ACCIDENT,
#     }.get(status, COLOR_NORMAL)

# def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
#     """Draw a label with background."""
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness, cv2.LINE_AA)

# def draw_status_banner(frame, status, count, max_iou, fps):
#     """Draw top status banner on frame."""
#     h, w = frame.shape[:2]
#     color = get_status_color(status)

#     # Semi-transparent banner
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     # Status indicator circle
#     cv2.circle(frame, (35, 35), 14, color, -1)
#     cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)

#     # Status text
#     icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
#     label = icons.get(status, status)
#     cv2.putText(frame, label, (60, 45),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

#     # Stats
#     stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w - 380, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     # Bottom bar: timestamp
#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     """Draw bounding boxes for each detected vehicle."""
#     color = get_status_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle Detection System  —  YOLO11")
#     print("  Status: NORMAL | CONGESTION | ACCIDENT")
#     print("=" * 55)
#     print(f"[INFO] Loading YOLO11 model...")

#     model = YOLO("yolo11n.pt")   # Downloads automatically on first run
#     print("[INFO] Model loaded. Opening webcam...")

#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press  Q  to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_iou    = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to read frame from webcam.")
#             break

#         frame_idx += 1

#         # ── FPS calculation ──────────────────
#         now = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         if elapsed > 0:
#             fps = 1.0 / elapsed
#         t_prev = now

#         # ── Detection (every FRAME_SKIP frames) ─
#         if frame_idx % FRAME_SKIP == 0:
#             results = model(frame, verbose=False)[0]

#             detections = []
#             boxes_raw  = []

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = VEHICLE_CLASSES[cls_id]
#                 detections.append(((x1, y1, x2, y2), label, conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Scene analysis ───────────────
#             status, max_iou, count = analyze_scene(boxes_raw)

#             # ── Console alert (on change) ────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 alert_icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                     "ACCIDENT":   "🚨",
#                 }
#                 print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
#                       f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, max_iou)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_iou    = max_iou

#         # ── Draw ────────────────────────────
#         if frame_idx % FRAME_SKIP == 0:
#             draw_boxes(frame, detections, last_status)

#         draw_status_banner(frame, last_status, last_count, last_iou, fps)

#         # ── Show ────────────────────────────
#         cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit requested by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved to: {LOG_FILE}")
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     main()




#==================================end of version 1.3==================================


# """
# Vehicle Detection System using YOLO11
# Detects: Normal | Congestion | Accident
# Input: Webcam (live camera)
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX         = 0          # Change to 1, 2... if multiple cameras
# CONGESTION_THRESHOLD = 6          # Vehicles in frame = congestion
# OVERLAP_IOU_THRESHOLD= 0.15       # IoU above this = possible accident
# ACCIDENT_MIN_BOXES   = 2          # Minimum overlapping vehicles for accident
# LOG_FILE             = "detection_log.csv"
# FRAME_SKIP           = 2          # Process every Nth frame (performance)

# # YOLO vehicle class IDs (COCO dataset)
# VEHICLE_CLASSES = {
#     2:  "car",
#     3:  "motorcycle",
#     5:  "bus",
#     7:  "truck",
# }

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)      # Green
# COLOR_CONGESTION = (0, 165, 255)    # Orange
# COLOR_ACCIDENT   = (0, 0, 220)      # Red
# COLOR_BOX        = (255, 220, 50)   # Yellow-ish box
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING SETUP
# # ─────────────────────────────────────────────
# def init_log(path):
#     """Initialize CSV log file with headers."""
#     file_exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     writer = csv.writer(f)
#     if not file_exists:
#         writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
#     return f, writer

# def log_event(writer, status, count, max_iou):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU (Intersection over Union)
# # ─────────────────────────────────────────────
# def compute_iou(boxA, boxB):
#     """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union_area = areaA + areaB - inter_area

#     return inter_area / union_area if union_area > 0 else 0.0

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(boxes):
#     """
#     Given a list of bounding boxes, return:
#       - status : 'NORMAL' | 'CONGESTION' | 'ACCIDENT'
#       - max_iou: highest pairwise IoU found
#     """
#     count = len(boxes)
#     max_iou = 0.0
#     overlap_count = 0

#     # Check pairwise IoU for accident detection
#     for i in range(count):
#         for j in range(i + 1, count):
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > max_iou:
#                 max_iou = iou
#             if iou >= OVERLAP_IOU_THRESHOLD:
#                 overlap_count += 1

#     # Decision logic
#     if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
#         status = "ACCIDENT"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, max_iou, count

# # ─────────────────────────────────────────────
# #  DRAWING HELPERS
# # ─────────────────────────────────────────────
# def get_status_color(status):
#     return {
#         "NORMAL":     COLOR_NORMAL,
#         "CONGESTION": COLOR_CONGESTION,
#         "ACCIDENT":   COLOR_ACCIDENT,
#     }.get(status, COLOR_NORMAL)

# def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
#     """Draw a label with background."""
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness, cv2.LINE_AA)

# def draw_status_banner(frame, status, count, max_iou, fps):
#     """Draw top status banner on frame."""
#     h, w = frame.shape[:2]
#     color = get_status_color(status)

#     # Semi-transparent banner
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     # Status indicator circle
#     cv2.circle(frame, (35, 35), 14, color, -1)
#     cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)

#     # Status text
#     icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
#     label = icons.get(status, status)
#     cv2.putText(frame, label, (60, 45),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

#     # Stats
#     stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w - 380, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     # Bottom bar: timestamp
#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     """Draw bounding boxes for each detected vehicle."""
#     color = get_status_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle Detection System  —  YOLO11")
#     print("  Status: NORMAL | CONGESTION | ACCIDENT")
#     print("=" * 55)
#     print(f"[INFO] Loading YOLO11 model...")

#     model = YOLO("yolo11n.pt")   # Downloads automatically on first run
#     print("[INFO] Model loaded. Opening webcam...")

#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press  Q  to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_iou    = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to read frame from webcam.")
#             break

#         frame_idx += 1

#         # ── FPS calculation ──────────────────
#         now = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         if elapsed > 0:
#             fps = 1.0 / elapsed
#         t_prev = now

#         # ── Detection (every FRAME_SKIP frames) ─
#         if frame_idx % FRAME_SKIP == 0:
#             results = model(frame, verbose=False)[0]

#             detections = []
#             boxes_raw  = []

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = VEHICLE_CLASSES[cls_id]
#                 detections.append(((x1, y1, x2, y2), label, conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Scene analysis ───────────────
#             status, max_iou, count = analyze_scene(boxes_raw)

#             # ── Console alert (on change) ────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 alert_icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                     "ACCIDENT":   "🚨",
#                 }
#                 print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
#                       f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, max_iou)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_iou    = max_iou

#         # ── Draw ────────────────────────────
#         if frame_idx % FRAME_SKIP == 0:
#             draw_boxes(frame, detections, last_status)

#         draw_status_banner(frame, last_status, last_count, last_iou, fps)

#         # ── Show ────────────────────────────
#         cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit requested by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved to: {LOG_FILE}")
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     main()




#==================================end of version 1.2==================================


# """
# Vehicle Detection System — YOLO11 + Custom Accident Model
# Uses TWO models:
#   1. yolo11n.pt         → detects vehicles (car, truck, bus, motorcycle)
#   2. best.pt (custom)   → classifies accident severity (Moderate / Severe)

# Input : Webcam
# Output: Screen display + console alerts + log file

# Fix v1.1: Accident detection now requires at least 1 vehicle in frame.
#           No vehicles = forced NORMAL, regardless of accident model output.

# Fix v1.2: Priority logic added to resolve congestion vs accident conflict.
#           - Accident requires confidence > 0.60 and vehicle count <= 8
#           - Congestion wins when both fire unless accident conf > 0.75
#           - Very high confidence accident (> 0.75) overrides congestion
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv
# from pathlib import Path

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX          = 0
# CONGESTION_THRESHOLD  = 6
# LOG_FILE              = "detection_log.csv"
# FRAME_SKIP            = 2

# # Accident detection thresholds
# ACCIDENT_CONF_MIN      = 0.60   # Minimum confidence to consider accident
# ACCIDENT_CONF_OVERRIDE = 0.75   # Confidence needed to override congestion
# ACCIDENT_MAX_VEHICLES  = 8      # Accidents unlikely above this vehicle count

# # Model paths
# VEHICLE_MODEL_PATH  = "yolo11n.pt"
# ACCIDENT_MODEL_PATH = r"runs\detect\accident_training\accident_v1\weights\best.pt"

# # YOLO COCO vehicle class IDs
# VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)
# COLOR_CONGESTION = (0, 165, 255)
# COLOR_MODERATE   = (0, 200, 255)
# COLOR_SEVERE     = (0, 0, 220)
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING
# # ─────────────────────────────────────────────
# def init_log(path):
#     exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     w = csv.writer(f)
#     if not exists:
#         w.writerow(["timestamp", "status", "vehicle_count", "accident_class", "confidence"])
#     return f, w

# def log_event(writer, status, count, acc_class, conf):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, acc_class, f"{conf:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU
# # ─────────────────────────────────────────────
# def compute_iou(a, b):
#     xA, yA = max(a[0], b[0]), max(a[1], b[1])
#     xB, yB = min(a[2], b[2]), min(a[3], b[3])
#     inter  = max(0, xB - xA) * max(0, yB - yA)
#     if inter == 0:
#         return 0.0
#     areaA = (a[2]-a[0]) * (a[3]-a[1])
#     areaB = (b[2]-b[0]) * (b[3]-b[1])
#     return inter / (areaA + areaB - inter)

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS  ← v1.2 PRIORITY LOGIC
# # ─────────────────────────────────────────────
# def analyze_scene(vehicle_boxes, accident_results):
#     """
#     Combine vehicle count + custom accident model results.
#     Returns: status, accident_class, confidence, count

#     Priority rules:
#       1. No vehicles          → always NORMAL
#       2. Accident conf > 0.75 → ACCIDENT overrides congestion
#       3. Both firing + conf < 0.75 → CONGESTION wins
#       4. Accident conf > 0.60 + vehicles <= 8 → ACCIDENT
#       5. Vehicles >= threshold → CONGESTION
#       6. Everything else      → NORMAL
#     """
#     count = len(vehicle_boxes)

#     # Rule 1 — Gate: no vehicles = no accident possible
#     if count == 0:
#         return "NORMAL", "none", 0.0, 0

#     # Check custom accident model output
#     accident_class = "none"
#     acc_confidence = 0.0

#     if accident_results and len(accident_results[0].boxes) > 0:
#         for box in accident_results[0].boxes:
#             conf     = float(box.conf[0])
#             cls_id   = int(box.cls[0])
#             cls_name = accident_results[0].names[cls_id]
#             if conf > acc_confidence:
#                 acc_confidence = conf
#                 accident_class = cls_name

#     # Evaluate conditions
#     is_accident = (
#         accident_class in ["Moderate", "Severe"]
#         and acc_confidence >= ACCIDENT_CONF_MIN
#         and count <= ACCIDENT_MAX_VEHICLES
#     )
#     is_congestion = count >= CONGESTION_THRESHOLD

#     # Rule 2 — Very high confidence accident overrides congestion
#     if is_accident and acc_confidence >= ACCIDENT_CONF_OVERRIDE:
#         status = f"ACCIDENT ({accident_class.upper()})"

#     # Rule 3 — Both firing but accident not confident enough → congestion wins
#     elif is_accident and is_congestion:
#         status = "CONGESTION"

#     # Rule 4 — Clear accident, no congestion
#     elif is_accident and not is_congestion:
#         status = f"ACCIDENT ({accident_class.upper()})"

#     # Rule 5 — Clear congestion, no accident signal
#     elif is_congestion:
#         status = "CONGESTION"

#     # Rule 6 — Nothing triggered
#     else:
#         status = "NORMAL"

#     return status, accident_class, acc_confidence, count

# # ─────────────────────────────────────────────
# #  DRAWING
# # ─────────────────────────────────────────────
# def get_color(status):
#     if "SEVERE" in status:     return COLOR_SEVERE
#     if "MODERATE" in status:   return COLOR_MODERATE
#     if "CONGESTION" in status: return COLOR_CONGESTION
#     return COLOR_NORMAL

# def draw_label(img, text, pos, color):
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

# def draw_banner(frame, status, count, acc_class, conf, fps):
#     h, w = frame.shape[:2]
#     color = get_color(status)

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     cv2.circle(frame, (35, 37), 15, color, -1)
#     cv2.circle(frame, (35, 37), 15, (255, 255, 255), 1)

#     icons = {
#         "NORMAL":     "✓ NORMAL",
#         "CONGESTION": "⚠ CONGESTION",
#     }
#     label = icons.get(status, f"✖ {status}")
#     cv2.putText(frame, label, (62, 50),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.05, color, 2, cv2.LINE_AA)

#     stats = f"Vehicles: {count}   Conf: {conf:.0%}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w-390, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     color = get_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle + Accident Detection — YOLO11 v1.2")
#     print("  Classes: Normal | Congestion | Moderate | Severe")
#     print("=" * 55)

#     # Load vehicle detection model
#     print("[INFO] Loading vehicle detection model...")
#     vehicle_model = YOLO(VEHICLE_MODEL_PATH)

#     # Load custom accident model
#     accident_model = None
#     if Path(ACCIDENT_MODEL_PATH).exists():
#         print("[INFO] Loading custom accident model...")
#         accident_model = YOLO(ACCIDENT_MODEL_PATH)
#         print("[INFO] ✅ Custom accident model loaded!")
#     else:
#         print(f"[WARN] ⚠️  Custom model not found at: {ACCIDENT_MODEL_PATH}")
#         print("[WARN]    Train first using train_accident_model.py")
#         print("[WARN]    Running with vehicle detection only...\n")

#     # Open webcam
#     print("[INFO] Opening webcam...")
#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print("[ERROR] Cannot open webcam!")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print(f"[INFO] Thresholds → Accident min conf : {ACCIDENT_CONF_MIN:.0%} "
#           f"| Override conf: {ACCIDENT_CONF_OVERRIDE:.0%} "
#           f"| Max vehicles for accident: {ACCIDENT_MAX_VEHICLES}")
#     print("[INFO] Press Q to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_conf   = 0.0
#     last_class  = "none"
#     detections  = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_idx += 1
#         now     = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         fps     = 1.0 / elapsed if elapsed > 0 else fps
#         t_prev  = now

#         if frame_idx % FRAME_SKIP == 0:

#             # ── Vehicle detection ────────────
#             v_results  = vehicle_model(frame, verbose=False)[0]
#             detections = []
#             boxes_raw  = []

#             for box in v_results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 detections.append(((x1, y1, x2, y2), VEHICLE_CLASSES[cls_id], conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Accident model ───────────────
#             acc_results = None
#             if accident_model:
#                 acc_results = accident_model(frame, verbose=False)

#             # ── Scene analysis ───────────────
#             status, acc_class, acc_conf, count = analyze_scene(boxes_raw, acc_results)

#             # ── Console alert ────────────────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                 }
#                 icon = icons.get(status, "🚨")
#                 print(f"[{ts}] {icon} STATUS → {status} "
#                       f"| Vehicles: {count} | Confidence: {acc_conf:.0%}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, acc_class, acc_conf)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_conf   = acc_conf
#             last_class  = acc_class

#         # ── Draw ────────────────────────────
#         draw_boxes(frame, detections, last_status)
#         draw_banner(frame, last_status, last_count, last_class, last_conf, fps)

#         cv2.imshow("Vehicle + Accident Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved: {LOG_FILE}")

# if __name__ == "__main__":
#     main()





# =========================== end of version 1.3 ===========================













# """
# Vehicle Detection System using YOLO11
# Detects: Normal | Congestion | Accident
# Input: Webcam (live camera)
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX         = 0          # Change to 1, 2... if multiple cameras
# CONGESTION_THRESHOLD = 6          # Vehicles in frame = congestion
# OVERLAP_IOU_THRESHOLD= 0.15       # IoU above this = possible accident
# ACCIDENT_MIN_BOXES   = 2          # Minimum overlapping vehicles for accident
# LOG_FILE             = "detection_log.csv"
# FRAME_SKIP           = 2          # Process every Nth frame (performance)

# # YOLO vehicle class IDs (COCO dataset)
# VEHICLE_CLASSES = {
#     2:  "car",
#     3:  "motorcycle",
#     5:  "bus",
#     7:  "truck",
# }

# # Colors (BGR)
# COLOR_NORMAL     = (0, 200, 0)      # Green
# COLOR_CONGESTION = (0, 165, 255)    # Orange
# COLOR_ACCIDENT   = (0, 0, 220)      # Red
# COLOR_BOX        = (255, 220, 50)   # Yellow-ish box
# COLOR_TEXT_BG    = (20, 20, 20)

# # ─────────────────────────────────────────────
# #  LOGGING SETUP
# # ─────────────────────────────────────────────
# def init_log(path):
#     """Initialize CSV log file with headers."""
#     file_exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     writer = csv.writer(f)
#     if not file_exists:
#         writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
#     return f, writer

# def log_event(writer, status, count, max_iou):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# # ─────────────────────────────────────────────
# #  IoU (Intersection over Union)
# # ─────────────────────────────────────────────
# def compute_iou(boxA, boxB):
#     """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return 0.0

#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union_area = areaA + areaB - inter_area

#     return inter_area / union_area if union_area > 0 else 0.0

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(boxes):
#     """
#     Given a list of bounding boxes, return:
#       - status : 'NORMAL' | 'CONGESTION' | 'ACCIDENT'
#       - max_iou: highest pairwise IoU found
#     """
#     count = len(boxes)
#     max_iou = 0.0
#     overlap_count = 0

#     # Check pairwise IoU for accident detection
#     for i in range(count):
#         for j in range(i + 1, count):
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > max_iou:
#                 max_iou = iou
#             if iou >= OVERLAP_IOU_THRESHOLD:
#                 overlap_count += 1

#     # Decision logic
#     if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
#         status = "ACCIDENT"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"

#     return status, max_iou, count

# # ─────────────────────────────────────────────
# #  DRAWING HELPERS
# # ─────────────────────────────────────────────
# def get_status_color(status):
#     return {
#         "NORMAL":     COLOR_NORMAL,
#         "CONGESTION": COLOR_CONGESTION,
#         "ACCIDENT":   COLOR_ACCIDENT,
#     }.get(status, COLOR_NORMAL)

# def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
#     """Draw a label with background."""
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness, cv2.LINE_AA)

# def draw_status_banner(frame, status, count, max_iou, fps):
#     """Draw top status banner on frame."""
#     h, w = frame.shape[:2]
#     color = get_status_color(status)

#     # Semi-transparent banner
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

#     # Status indicator circle
#     cv2.circle(frame, (35, 35), 14, color, -1)
#     cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)

#     # Status text
#     icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
#     label = icons.get(status, status)
#     cv2.putText(frame, label, (60, 45),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

#     # Stats
#     stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w - 380, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

#     # Bottom bar: timestamp
#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     """Draw bounding boxes for each detected vehicle."""
#     color = get_status_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# # ─────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────
# def main():
#     print("=" * 55)
#     print("  Vehicle Detection System  —  YOLO11")
#     print("  Status: NORMAL | CONGESTION | ACCIDENT")
#     print("=" * 55)
#     print(f"[INFO] Loading YOLO11 model...")

#     model = YOLO("yolo11n.pt")   # Downloads automatically on first run
#     print("[INFO] Model loaded. Opening webcam...")

#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}. Try changing WEBCAM_INDEX.")
#         return

#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     log_f, log_writer = init_log(LOG_FILE)
#     print(f"[INFO] Logging to: {LOG_FILE}")
#     print("[INFO] Press  Q  to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_iou    = 0.0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to read frame from webcam.")
#             break

#         frame_idx += 1

#         # ── FPS calculation ──────────────────
#         now = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         if elapsed > 0:
#             fps = 1.0 / elapsed
#         t_prev = now

#         # ── Detection (every FRAME_SKIP frames) ─
#         if frame_idx % FRAME_SKIP == 0:
#             results = model(frame, verbose=False)[0]

#             detections = []
#             boxes_raw  = []

#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = VEHICLE_CLASSES[cls_id]
#                 detections.append(((x1, y1, x2, y2), label, conf))
#                 boxes_raw.append([x1, y1, x2, y2])

#             # ── Scene analysis ───────────────
#             status, max_iou, count = analyze_scene(boxes_raw)

#             # ── Console alert (on change) ────
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 alert_icons = {
#                     "NORMAL":     "✅",
#                     "CONGESTION": "🚦",
#                     "ACCIDENT":   "🚨",
#                 }
#                 print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
#                       f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")

#             # ── Log every 30 frames ──────────
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, max_iou)
#                 log_f.flush()

#             last_status = status
#             last_count  = count
#             last_iou    = max_iou

#         # ── Draw ────────────────────────────
#         if frame_idx % FRAME_SKIP == 0:
#             draw_boxes(frame, detections, last_status)

#         draw_status_banner(frame, last_status, last_count, last_iou, fps)

#         # ── Show ────────────────────────────
#         cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             print("\n[INFO] Quit requested by user.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()
#     print(f"[INFO] Log saved to: {LOG_FILE}")
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     main()




#==================================end of version 1.0==================================


# """
# Vehicle Detection System using YOLO11
# Detects: Normal | Congestion | Accident
# Input: Webcam (live camera)
# Output: Screen display + console alerts + log file
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import os
# import csv

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────
# WEBCAM_INDEX         = 0
# CONGESTION_THRESHOLD = 6
# OVERLAP_IOU_THRESHOLD= 0.15
# ACCIDENT_MIN_BOXES   = 2
# LOG_FILE             = "detection_log.csv"
# FRAME_SKIP           = 2

# VEHICLE_CLASSES = {
#     2:  "car",
#     3:  "motorcycle",
#     5:  "bus",
#     7:  "truck",
# }

# COLOR_NORMAL     = (0, 200, 0)
# COLOR_CONGESTION = (0, 165, 255)
# COLOR_ACCIDENT   = (0, 0, 220)
# COLOR_BOX        = (255, 220, 50)
# COLOR_TEXT_BG    = (20, 20, 20)

# def init_log(path):
#     file_exists = os.path.isfile(path)
#     f = open(path, "a", newline="")
#     writer = csv.writer(f)
#     if not file_exists:
#         writer.writerow(["timestamp", "status", "vehicle_count", "max_iou"])
#     return f, writer

# def log_event(writer, status, count, max_iou):
#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     writer.writerow([ts, status, count, f"{max_iou:.3f}"])

# def compute_iou(boxA, boxB):
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])
#     inter_w = max(0, xB - xA)
#     inter_h = max(0, yB - yA)
#     inter_area = inter_w * inter_h
#     if inter_area == 0:
#         return 0.0
#     areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#     union_area = areaA + areaB - inter_area
#     return inter_area / union_area if union_area > 0 else 0.0

# def analyze_scene(boxes):
#     count = len(boxes)
#     max_iou = 0.0
#     overlap_count = 0
#     for i in range(count):
#         for j in range(i + 1, count):
#             iou = compute_iou(boxes[i], boxes[j])
#             if iou > max_iou:
#                 max_iou = iou
#             if iou >= OVERLAP_IOU_THRESHOLD:
#                 overlap_count += 1
#     if overlap_count >= ACCIDENT_MIN_BOXES and max_iou >= OVERLAP_IOU_THRESHOLD:
#         status = "ACCIDENT"
#     elif count >= CONGESTION_THRESHOLD:
#         status = "CONGESTION"
#     else:
#         status = "NORMAL"
#     return status, max_iou, count

# def get_status_color(status):
#     return {
#         "NORMAL":     COLOR_NORMAL,
#         "CONGESTION": COLOR_CONGESTION,
#         "ACCIDENT":   COLOR_ACCIDENT,
#     }.get(status, COLOR_NORMAL)

# def draw_label(img, text, pos, color, font_scale=0.55, thickness=1):
#     x, y = pos
#     (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#     cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
#     cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX,
#                 font_scale, color, thickness, cv2.LINE_AA)

# def draw_status_banner(frame, status, count, max_iou, fps):
#     h, w = frame.shape[:2]
#     color = get_status_color(status)
#     overlay = frame.copy()
#     cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
#     cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
#     cv2.circle(frame, (35, 35), 14, color, -1)
#     cv2.circle(frame, (35, 35), 14, (255, 255, 255), 1)
#     icons = {"NORMAL": "✓ NORMAL", "CONGESTION": "⚠ CONGESTION", "ACCIDENT": "✖ ACCIDENT"}
#     label = icons.get(status, status)
#     cv2.putText(frame, label, (60, 45),
#                 cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)
#     stats = f"Vehicles: {count}   Max IoU: {max_iou:.2f}   FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w - 380, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

# def draw_boxes(frame, detections, status):
#     color = get_status_color(status)
#     for (x1, y1, x2, y2), label, conf in detections:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# def main():
#     print("=" * 55)
#     print("  Vehicle Detection System  —  YOLO11")
#     print("  Status: NORMAL | CONGESTION | ACCIDENT")
#     print("=" * 55)
#     model = YOLO("yolo11n.pt")
#     cap = cv2.VideoCapture(WEBCAM_INDEX)
#     if not cap.isOpened():
#         print(f"[ERROR] Cannot open webcam index {WEBCAM_INDEX}.")
#         return
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     log_f, log_writer = init_log(LOG_FILE)
#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     last_count  = 0
#     last_iou    = 0.0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_idx += 1
#         now = datetime.now()
#         elapsed = (now - t_prev).total_seconds()
#         if elapsed > 0:
#             fps = 1.0 / elapsed
#         t_prev = now
#         if frame_idx % FRAME_SKIP == 0:
#             results = model(frame, verbose=False)[0]
#             detections = []
#             boxes_raw  = []
#             for box in results.boxes:
#                 cls_id = int(box.cls[0])
#                 if cls_id not in VEHICLE_CLASSES:
#                     continue
#                 conf = float(box.conf[0])
#                 if conf < 0.35:
#                     continue
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 label = VEHICLE_CLASSES[cls_id]
#                 detections.append(((x1, y1, x2, y2), label, conf))
#                 boxes_raw.append([x1, y1, x2, y2])
#             status, max_iou, count = analyze_scene(boxes_raw)
#             if status != last_status:
#                 ts = datetime.now().strftime("%H:%M:%S")
#                 alert_icons = {"NORMAL": "✅", "CONGESTION": "🚦", "ACCIDENT": "🚨"}
#                 print(f"[{ts}] {alert_icons.get(status, '')} STATUS → {status} "
#                       f"| Vehicles: {count} | Max IoU: {max_iou:.3f}")
#             if frame_idx % 30 == 0:
#                 log_event(log_writer, status, count, max_iou)
#                 log_f.flush()
#             last_status = status
#             last_count  = count
#             last_iou    = max_iou
#         if frame_idx % FRAME_SKIP == 0:
#             draw_boxes(frame, detections, last_status)
#         draw_status_banner(frame, last_status, last_count, last_iou, fps)
#         cv2.imshow("Vehicle Detection — YOLO11  (Q to quit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     log_f.close()

# if __name__ == "__main__":
#     main()


#==================================end of version 1.0==================================


"""
Vehicle Detection System — YOLO11 + Custom Accident Model
Uses TWO models:
  1. yolo11n.pt         → detects vehicles (car, truck, bus, motorcycle)
  2. best.pt (custom)   → classifies accident severity (Moderate / Severe)

Input : ESP32-CAM MJPEG stream
Output: Screen display + console alerts + log file

Fix v1.1: No vehicles = forced NORMAL
Fix v1.2: Priority logic — congestion vs accident conflict resolved
v1.3    : Replaced webcam with ESP32-CAM stream + auto-reconnect
v1.4    : Fixed stream URL to http://192.168.8.122/stream
"""
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
import csv
import time
from pathlib import Path
import urllib.request

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ESP32_STREAM_URL = "http://192.168.8.122/capture"
RECONNECT_DELAY  = 2

CONGESTION_THRESHOLD = 6
LOG_FILE = "detection_log.csv"
FRAME_SKIP = 2

ACCIDENT_CONF_MIN      = 0.60
ACCIDENT_CONF_OVERRIDE = 0.75
ACCIDENT_MAX_VEHICLES  = 8

VEHICLE_MODEL_PATH  = "yolo11n.pt"
ACCIDENT_MODEL_PATH = r"runs\detect\accident_training\accident_v1\weights\best.pt"

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

COLOR_NORMAL     = (0, 200, 0)
COLOR_CONGESTION = (0, 165, 255)
COLOR_MODERATE   = (0, 200, 255)
COLOR_SEVERE     = (0, 0, 220)
COLOR_TEXT_BG    = (20, 20, 20)

# ─────────────────────────────────────────────
# ESP32 FRAME FETCH
# ─────────────────────────────────────────────
def get_frame(url):
    try:
        img_resp = urllib.request.urlopen(url, timeout=5)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, -1)
        frame = cv2.resize(frame, (640, 480))
        return True, frame
    except:
        return False, None

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def init_log(path):
    exists = os.path.isfile(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if not exists:
        w.writerow(["timestamp", "status", "vehicle_count", "accident_class", "confidence"])
    return f, w

def log_event(writer, status, count, acc_class, conf):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow([ts, status, count, acc_class, f"{conf:.3f}"])

# ─────────────────────────────────────────────
# SCENE ANALYSIS
# ─────────────────────────────────────────────
def analyze_scene(vehicle_boxes, accident_results):
    count = len(vehicle_boxes)

    if count == 0:
        return "NORMAL", "none", 0.0, 0

    accident_class = "none"
    acc_confidence = 0.0

    if accident_results and len(accident_results[0].boxes) > 0:
        for box in accident_results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = accident_results[0].names[cls_id]
            if conf > acc_confidence:
                acc_confidence = conf
                accident_class = cls_name

    is_accident = (
        accident_class in ["Moderate", "Severe"]
        and acc_confidence >= ACCIDENT_CONF_MIN
        and count <= ACCIDENT_MAX_VEHICLES
    )
    is_congestion = count >= CONGESTION_THRESHOLD

    if is_accident and acc_confidence >= ACCIDENT_CONF_OVERRIDE:
        status = f"ACCIDENT ({accident_class.upper()})"
    elif is_accident and is_congestion:
        status = "CONGESTION"
    elif is_accident:
        status = f"ACCIDENT ({accident_class.upper()})"
    elif is_congestion:
        status = "CONGESTION"
    else:
        status = "NORMAL"

    return status, accident_class, acc_confidence, count

# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────
def get_color(status):
    if "SEVERE" in status: return COLOR_SEVERE
    if "MODERATE" in status: return COLOR_MODERATE
    if "CONGESTION" in status: return COLOR_CONGESTION
    return COLOR_NORMAL

def draw_label(img, text, pos, color):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

def draw_banner(frame, status, count, acc_class, conf, fps):
    h, w = frame.shape[:2]
    color = get_color(status)

    cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)

    cv2.putText(frame, status, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    stats = f"Vehicles: {count} | Conf: {conf:.0%} | FPS: {fps:.1f}"
    cv2.putText(frame, stats, (w-350, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

def draw_boxes(frame, detections, status):
    color = get_color(status)
    for (x1, y1, x2, y2), label, conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, f"{label} {conf:.0%}", (x1, y1 - 2), color)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("🚀 Vehicle + Accident Detection (Stable ESP32 Version)")

    vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    accident_model = None
    if Path(ACCIDENT_MODEL_PATH).exists():
        accident_model = YOLO(ACCIDENT_MODEL_PATH)

    log_f, log_writer = init_log(LOG_FILE)

    frame_idx = 0
    t_prev = datetime.now()
    fps = 0

    last_status = "NORMAL"
    detections = []

    # ✅ FIXED VARIABLES
    count = 0
    acc_class = "none"
    acc_conf = 0.0

    while True:
        ret, frame = get_frame(ESP32_STREAM_URL)

        if not ret:
            print("[WARN] Camera fetch failed... retrying")
            time.sleep(RECONNECT_DELAY)
            continue

        frame_idx += 1

        now = datetime.now()
        fps = 1.0 / ((now - t_prev).total_seconds() + 1e-5)
        t_prev = now

        if frame_idx % FRAME_SKIP == 0:

            v_results = vehicle_model(frame, verbose=False)[0]
            detections = []
            boxes_raw = []

            for box in v_results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                conf = float(box.conf[0])
                if conf < 0.35:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(((x1, y1, x2, y2), VEHICLE_CLASSES[cls_id], conf))
                boxes_raw.append([x1, y1, x2, y2])

            acc_results = accident_model(frame, verbose=False) if accident_model else None

            last_status, acc_class, acc_conf, count = analyze_scene(boxes_raw, acc_results)

            if frame_idx % 30 == 0:
                log_event(log_writer, last_status, count, acc_class, acc_conf)
                log_f.flush()

        draw_boxes(frame, detections, last_status)
        draw_banner(frame, last_status, count, acc_class, acc_conf, fps)

        cv2.imshow("ESP32-CAM Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()
    log_f.close()

if __name__ == "__main__":
    main()