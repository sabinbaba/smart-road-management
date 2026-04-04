"""
main.py — FastAPI Backend
Smart Traffic Detection System

Run with:
    cd backend
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API Docs:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import urllib.request
import csv
import io
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

# Allow frontend (HTML files) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Initialize database on startup
@app.on_event("startup")
def startup():
    db.init_db()
    print("[API] ✅ Smart Traffic API is running!")
    print("[API] 📖 Docs at: http://localhost:8000/docs")

# ─────────────────────────────────────────────
#  REQUEST / RESPONSE MODELS
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
    accident_class : str  = "none"
    confidence     : float = 0.0

# ─────────────────────────────────────────────
#  AUTH HELPER
# ─────────────────────────────────────────────
def get_current_user(request: Request) -> dict:
    """Extract and verify token from Authorization header."""
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
    """Register a new user."""
    # Validate inputs
    errors = auth.validate_signup(body.username, body.email, body.password)
    if errors:
        raise HTTPException(status_code=400, detail=errors)

    # Check if user exists
    if db.get_user_by_email(body.email):
        raise HTTPException(status_code=400, detail="Email already registered.")
    if db.get_user_by_username(body.username):
        raise HTTPException(status_code=400, detail="Username already taken.")

    # Create user
    hashed = auth.hash_password(body.password)
    user_id = db.create_user(body.username, body.email, hashed)

    # Generate token
    token = auth.create_token(user_id, body.username, "user")

    return {
        "message" : "Account created successfully!",
        "token"   : token,
        "username": body.username,
        "role"    : "user"
    }

@app.post("/auth/login")
def login(body: LoginRequest):
    """Login and get token."""
    user = db.get_user_by_email(body.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if not auth.verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = auth.create_token(user["id"], user["username"], user["role"])

    return {
        "message" : "Login successful!",
        "token"   : token,
        "username": user["username"],
        "role"    : user["role"]
    }

@app.get("/auth/me")
def get_me(user = Depends(get_current_user)):
    """Get current logged in user info."""
    return {
        "user_id" : user["sub"],
        "username": user["username"],
        "role"    : user["role"]
    }

# ─────────────────────────────────────────────
#  DETECTION ROUTES
# ─────────────────────────────────────────────
@app.post("/api/detections")
def save_detection(body: DetectionRequest):
    """
    Save a new detection from vehicle_detector.py.
    Called automatically by the detector every 30 frames.
    """
    detection_id = db.insert_detection(
        status         = body.status,
        vehicle_count  = body.vehicle_count,
        accident_class = body.accident_class,
        confidence     = body.confidence
    )
    return {"id": detection_id, "message": "Detection saved."}

@app.get("/api/status")
def get_status():
    """Get the latest detection status — polled every few seconds by frontend."""
    latest = db.get_latest_detection()
    if not latest:
        return {
            "status"        : "NORMAL",
            "vehicle_count" : 0,
            "accident_class": "none",
            "confidence"    : 0.0,
            "timestamp"     : None
        }
    return latest

@app.get("/api/detections")
def get_detections(
    page          : int            = Query(1, ge=1),
    per_page      : int            = Query(20, ge=1, le=100),
    status_filter : Optional[str]  = Query(None),
    date_filter   : Optional[str]  = Query(None),
):
    """Get paginated detection history with optional filters."""
    return db.get_detections(
        page          = page,
        per_page      = per_page,
        status_filter = status_filter,
        date_filter   = date_filter
    )

@app.get("/api/detections/today")
def get_today():
    """Get all detections from today."""
    return {"data": db.get_today_detections()}

# ─────────────────────────────────────────────
#  STATS / ANALYTICS ROUTES
# ─────────────────────────────────────────────
@app.get("/api/stats/summary")
def get_summary():
    """Get overall summary statistics for dashboard cards."""
    return db.get_summary_stats()

@app.get("/api/stats/hourly")
def get_hourly():
    """Get hourly breakdown for today — used by analytics charts."""
    return {"data": db.get_hourly_stats()}

@app.get("/api/stats/daily")
def get_daily(days: int = Query(7, ge=1, le=30)):
    """Get daily stats for the last N days — used by trend charts."""
    return {"data": db.get_daily_stats_range(days)}

# ─────────────────────────────────────────────
#  ESP32-CAM STREAM PROXY
# ─────────────────────────────────────────────
ESP32_STREAM_URL = "http://192.168.8.122/stream"

@app.get("/api/stream")
def proxy_stream():
    """
    Proxy the ESP32-CAM MJPEG stream to the frontend.
    Frontend can embed this as: <img src="http://localhost:8000/api/stream">
    """
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
#  REPORT DOWNLOAD
# ─────────────────────────────────────────────
@app.get("/api/report/csv")
def download_csv(
    date_filter: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
):
    """Download detections as CSV report."""
    result = db.get_detections(
        page          = 1,
        per_page      = 10000,
        status_filter = status_filter,
        date_filter   = date_filter
    )
    rows = result["data"]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id", "timestamp", "status",
        "vehicle_count", "accident_class", "confidence"
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
    """Get a summary report for the last N days."""
    daily  = db.get_daily_stats_range(days)
    summary = db.get_summary_stats()

    return {
        "generated_at" : datetime.now().isoformat(),
        "period_days"  : days,
        "summary"      : summary,
        "daily_breakdown": daily
    }

# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
