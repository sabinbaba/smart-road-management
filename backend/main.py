"""
main.py — FastAPI Backend
Smart Traffic Detection System

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import urllib.request
import asyncio
import json
import csv
import io
import queue
import threading
from datetime import datetime

import database as db
import auth

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title       = "Smart Traffic Detection API",
    description = "Backend API for Vehicle + Accident Detection System",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

@app.on_event("startup")
def startup():
    db.init_db()
    print("[API] ✅ Smart Traffic API is running!")
    print("[API] 📡 SSE endpoint: http://localhost:8000/api/events")
    print("[API] 📖 Docs at:      http://localhost:8000/docs")

# ─────────────────────────────────────────────
#  SSE — REAL-TIME EVENT BUS
#  vehicle_detector.py POSTs to /api/detections
#  frontend connects to /api/events and gets
#  pushed instantly — no polling needed
# ─────────────────────────────────────────────

# Thread-safe queue for SSE clients
_sse_clients: list[asyncio.Queue] = []
_sse_lock = threading.Lock()

# Latest detection state (for new clients joining)
_latest_detection = {
    "status":         "NORMAL",
    "vehicle_count":  0,
    "accident_class": "none",
    "confidence":     0.0,
    "timestamp":      None,
}

def push_event_to_clients(data: dict):
    """Push a detection event to all connected SSE clients."""
    global _latest_detection
    _latest_detection = data
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            _sse_clients.remove(q)

# ─────────────────────────────────────────────
#  REQUEST MODELS
# ─────────────────────────────────────────────
class SignupRequest(BaseModel):
    username : str
    email    : str
    password : str

class LoginRequest(BaseModel):
    email    : str
    password : str

class DetectionRequest(BaseModel):
    status         : str
    vehicle_count  : int
    accident_class : str   = "none"
    confidence     : float = 0.0

# ─────────────────────────────────────────────
#  AUTH HELPER
# ─────────────────────────────────────────────
def get_current_user(request: Request) -> dict:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header.split(" ")[1]
    payload = auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

# ─────────────────────────────────────────────
#  ROOT
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Smart Traffic Detection API",
        "version": "1.0.0",
        "status" : "running",
        "docs"   : "/docs"
    }

# ─────────────────────────────────────────────
#  AUTH ROUTES
# ─────────────────────────────────────────────
@app.post("/auth/signup")
def signup(body: SignupRequest):
    errors = auth.validate_signup(body.username, body.email, body.password)
    if errors:
        raise HTTPException(status_code=400, detail=errors)
    if db.get_user_by_email(body.email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    if db.get_user_by_username(body.username):
        raise HTTPException(status_code=400, detail="Username already taken.")
    hashed  = auth.hash_password(body.password)
    user_id = db.create_user(body.username, body.email, hashed)
    token   = auth.create_token(user_id, body.username, "user")
    return {"message": "Account created!", "token": token,
            "username": body.username, "role": "user"}

@app.post("/auth/login")
def login(body: LoginRequest):
    user = db.get_user_by_email(body.email)
    if not user or not auth.verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = auth.create_token(user["id"], user["username"], user["role"])
    return {"message": "Login successful!", "token": token,
            "username": user["username"], "role": user["role"]}

@app.get("/auth/me")
def get_me(user=Depends(get_current_user)):
    return {"user_id": user["sub"], "username": user["username"], "role": user["role"]}

# ─────────────────────────────────────────────
#  ★ DETECTION — POST from vehicle_detector.py
# ─────────────────────────────────────────────
@app.post("/api/detections")
def save_detection(body: DetectionRequest):
    """
    Called by vehicle_detector.py every N frames.
    Saves to SQLite AND pushes to all SSE clients instantly.
    """
    detection_id = db.insert_detection(
        status         = body.status,
        vehicle_count  = body.vehicle_count,
        accident_class = body.accident_class,
        confidence     = body.confidence
    )

    # Build event payload
    event_data = {
        "id"            : detection_id,
        "status"        : body.status,
        "vehicle_count" : body.vehicle_count,
        "accident_class": body.accident_class,
        "confidence"    : round(body.confidence, 4),
        "timestamp"     : datetime.now().isoformat(),
    }

    # ★ Push to all connected SSE clients immediately
    push_event_to_clients(event_data)

    return {"id": detection_id, "message": "Detection saved and pushed."}

# ─────────────────────────────────────────────
#  ★ SSE — Frontend connects here for real-time
# ─────────────────────────────────────────────
@app.get("/api/events")
async def sse_stream(request: Request):
    """
    Server-Sent Events endpoint.
    Frontend connects once and receives pushed events instantly
    whenever vehicle_detector.py sends a new detection.

    Usage in frontend:
        const es = new EventSource('http://localhost:8000/api/events');
        es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            // update UI with data.status, data.vehicle_count etc
        };
    """
    client_queue: asyncio.Queue = asyncio.Queue(maxsize=50)

    with _sse_lock:
        _sse_clients.append(client_queue)

    async def event_generator():
        # Send latest state immediately on connect
        yield f"data: {json.dumps(_latest_detection)}\n\n"

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # Wait for next event (timeout so we can check disconnect)
                    data = await asyncio.wait_for(
                        client_queue.get(), timeout=3.0
                    )
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield ": heartbeat\n\n"

        except (GeneratorExit, asyncio.CancelledError):
            pass
        finally:
            with _sse_lock:
                if client_queue in _sse_clients:
                    _sse_clients.remove(client_queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )

# ─────────────────────────────────────────────
#  STATUS (fallback for pages that still poll)
# ─────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    """Returns latest detection. Used as SSE fallback."""
    latest = db.get_latest_detection()
    if not latest:
        return _latest_detection
    return latest

# ─────────────────────────────────────────────
#  DETECTIONS
# ─────────────────────────────────────────────
@app.get("/api/detections")
def get_detections(
    page          : int           = Query(1, ge=1),
    per_page      : int           = Query(20, ge=1, le=100),
    status_filter : Optional[str] = Query(None),
    date_filter   : Optional[str] = Query(None),
):
    return db.get_detections(
        page=page, per_page=per_page,
        status_filter=status_filter, date_filter=date_filter
    )

@app.get("/api/detections/today")
def get_today():
    return {"data": db.get_today_detections()}

# ─────────────────────────────────────────────
#  STATS
# ─────────────────────────────────────────────
@app.get("/api/stats/summary")
def get_summary():
    return db.get_summary_stats()

@app.get("/api/stats/hourly")
def get_hourly():
    return {"data": db.get_hourly_stats()}

@app.get("/api/stats/daily")
def get_daily(days: int = Query(7, ge=1, le=30)):
    return {"data": db.get_daily_stats_range(days)}

# ─────────────────────────────────────────────
#  ESP32-CAM STREAM PROXY
# ─────────────────────────────────────────────
ESP32_STREAM_URL = "http://192.168.8.122/stream"

@app.get("/api/stream")
def proxy_stream():
    def generate():
        try:
            stream = urllib.request.urlopen(ESP32_STREAM_URL, timeout=10)
            while True:
                chunk = stream.read(1024)
                if not chunk:
                    break
                yield chunk
        except Exception as e:
            print(f"[STREAM] Error: {e}")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ─────────────────────────────────────────────
#  REPORTS / EXPORT
# ─────────────────────────────────────────────
@app.get("/api/report/csv")
def download_csv(
    date_filter   : Optional[str] = Query(None),
    status_filter : Optional[str] = Query(None),
):
    result = db.get_detections(page=1, per_page=10000,
        status_filter=status_filter, date_filter=date_filter)
    rows = result["data"]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id","timestamp","status","vehicle_count","accident_class","confidence"
    ])
    writer.writeheader()
    writer.writerows(rows)

    filename = f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        io.StringIO(output.getvalue()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/api/report/summary")
def get_report_summary(days: int = Query(7, ge=1, le=30)):
    return {
        "generated_at"   : datetime.now().isoformat(),
        "period_days"    : days,
        "summary"        : db.get_summary_stats(),
        "daily_breakdown": db.get_daily_stats_range(days)
    }

# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)