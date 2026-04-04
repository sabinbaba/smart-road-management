# """
# vehicle_detector.py — YOLO11 + ESP32-CAM + FastAPI backend
# v2.0: Now sends detections to backend API (SQLite) instead of CSV log

# Fix v1.1: No vehicles = forced NORMAL
# Fix v1.2: Priority logic — congestion vs accident conflict resolved
# v1.3    : Replaced webcam with ESP32-CAM stream + auto-reconnect
# v1.4    : Fixed stream URL to http://192.168.8.122/stream
# v2.0    : Detections now POST to FastAPI backend → SQLite
# """

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from datetime import datetime
# import time
# import requests
# import urllib.request
# from pathlib import Path

# # ─────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────

# # ESP32-CAM
# ESP32_STREAM_URL  = "http://192.168.8.122/capture"
# RECONNECT_DELAY   = 2

# # Backend API
# API_BASE_URL      = "http://localhost:8000"
# API_DETECTION_URL = f"{API_BASE_URL}/api/detections"

# # Detection settings
# CONGESTION_THRESHOLD   = 6
# FRAME_SKIP             = 2
# SAVE_EVERY_N_FRAMES    = 30   # POST to backend every N frames

# # Accident thresholds
# ACCIDENT_CONF_MIN      = 0.60
# ACCIDENT_CONF_OVERRIDE = 0.75
# ACCIDENT_MAX_VEHICLES  = 8

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
# #  ESP32 FRAME FETCH
# # ─────────────────────────────────────────────
# def get_frame(url):
#     try:
#         img_resp = urllib.request.urlopen(url, timeout=5)
#         img_np   = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         frame    = cv2.imdecode(img_np, -1)
#         frame    = cv2.resize(frame, (640, 480))
#         return True, frame
#     except:
#         return False, None

# # ─────────────────────────────────────────────
# #  BACKEND API
# # ─────────────────────────────────────────────
# def send_detection(status, vehicle_count, accident_class, confidence):
#     """POST detection data to FastAPI backend."""
#     try:
#         response = requests.post(
#             API_DETECTION_URL,
#             json={
#                 "status"         : status,
#                 "vehicle_count"  : vehicle_count,
#                 "accident_class" : accident_class,
#                 "confidence"     : round(confidence, 4)
#             },
#             timeout=2
#         )
#         return response.status_code == 200
#     except requests.exceptions.ConnectionError:
#         print("[API] ❌ Backend not reachable. Is the server running?")
#         return False
#     except Exception as e:
#         print(f"[API] ❌ Error: {e}")
#         return False

# def check_backend():
#     """Check if backend API is reachable."""
#     try:
#         r = requests.get(f"{API_BASE_URL}/", timeout=3)
#         return r.status_code == 200
#     except:
#         return False

# # ─────────────────────────────────────────────
# #  SCENE ANALYSIS
# # ─────────────────────────────────────────────
# def analyze_scene(vehicle_boxes, accident_results):
#     count = len(vehicle_boxes)

#     if count == 0:
#         return "NORMAL", "none", 0.0, 0

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

#     is_accident = (
#         accident_class in ["Moderate", "Severe"]
#         and acc_confidence >= ACCIDENT_CONF_MIN
#         and count <= ACCIDENT_MAX_VEHICLES
#     )
#     is_congestion = count >= CONGESTION_THRESHOLD

#     if is_accident and acc_confidence >= ACCIDENT_CONF_OVERRIDE:
#         status = f"ACCIDENT ({accident_class.upper()})"
#     elif is_accident and is_congestion:
#         status = "CONGESTION"
#     elif is_accident:
#         status = f"ACCIDENT ({accident_class.upper()})"
#     elif is_congestion:
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
#     cv2.putText(img, text, (x+3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

# def draw_banner(frame, status, count, acc_class, conf, fps, api_ok):
#     h, w = frame.shape[:2]
#     color = get_color(status)

#     cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
#     cv2.putText(frame, status, (20, 45),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

#     stats = f"Vehicles: {count} | Conf: {conf:.0%} | FPS: {fps:.1f}"
#     cv2.putText(frame, stats, (w-350, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

#     api_color = (0, 200, 0) if api_ok else (0, 0, 220)
#     api_label = "API: ONLINE" if api_ok else "API: OFFLINE"
#     cv2.putText(frame, api_label, (w-350, 55),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, api_color, 1)

#     ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
#     cv2.putText(frame, ts, (10, h-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

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
#     print("  Vehicle + Accident Detection — YOLO11 v2.0")
#     print("  Input  : ESP32-CAM Stream")
#     print("  Output : FastAPI Backend → SQLite")
#     print("=" * 55)

#     # Check backend
#     print(f"\n[INFO] Checking backend at {API_BASE_URL}...")
#     api_ok = check_backend()
#     if api_ok:
#         print("[INFO] ✅ Backend is online!")
#     else:
#         print("[WARN] ⚠️  Backend offline — detections won't be saved!")
#         print("[WARN]    Start it with: uvicorn main:app --reload")

#     # Load models
#     print("\n[INFO] Loading vehicle detection model...")
#     vehicle_model = YOLO(VEHICLE_MODEL_PATH)

#     accident_model = None
#     if Path(ACCIDENT_MODEL_PATH).exists():
#         print("[INFO] Loading custom accident model...")
#         accident_model = YOLO(ACCIDENT_MODEL_PATH)
#         print("[INFO] ✅ Accident model loaded!")
#     else:
#         print(f"[WARN] ⚠️  Accident model not found. Vehicle detection only.")

#     print(f"\n[INFO] ESP32-CAM  : {ESP32_STREAM_URL}")
#     print(f"[INFO] Backend API: {API_BASE_URL}")
#     print("[INFO] Press Q to quit.\n")

#     frame_idx   = 0
#     fps         = 0.0
#     t_prev      = datetime.now()
#     last_status = "NORMAL"
#     detections  = []
#     count       = 0
#     acc_class   = "none"
#     acc_conf    = 0.0

#     while True:
#         ret, frame = get_frame(ESP32_STREAM_URL)

#         if not ret:
#             print("[WARN] Camera fetch failed... retrying")
#             time.sleep(RECONNECT_DELAY)
#             continue

#         frame_idx += 1
#         now     = datetime.now()
#         fps     = 1.0 / ((now - t_prev).total_seconds() + 1e-5)
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
#             acc_results = accident_model(frame, verbose=False) if accident_model else None

#             # ── Scene analysis ───────────────
#             last_status, acc_class, acc_conf, count = analyze_scene(boxes_raw, acc_results)

#             # ── Send to backend every N frames ──
#             if frame_idx % SAVE_EVERY_N_FRAMES == 0:
#                 api_ok = send_detection(last_status, count, acc_class, acc_conf)
#                 ts     = datetime.now().strftime("%H:%M:%S")
#                 icons  = {"NORMAL": "✅", "CONGESTION": "🚦"}
#                 icon   = icons.get(last_status, "🚨")
#                 saved  = "💾 saved" if api_ok else "❌ not saved"
#                 print(f"[{ts}] {icon} {last_status} | Vehicles: {count} "
#                       f"| Conf: {acc_conf:.0%} | {saved}")

#         # ── Draw ────────────────────────────
#         draw_boxes(frame, detections, last_status)
#         draw_banner(frame, last_status, count, acc_class, acc_conf, fps, api_ok)

#         cv2.imshow("Vehicle + Accident Detection v2.0  (Q to quit)", frame)

#         if cv2.waitKey(1) == ord("q"):
#             print("\n[INFO] Quit by user.")
#             break

#     cv2.destroyAllWindows()
#     print("[INFO] Done.")

# if __name__ == "__main__":
#     main()




#======================backup========================