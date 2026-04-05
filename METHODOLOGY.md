# Smart Traffic Detection System — Methodology & Technology Stack

## Introduction

The Smart Traffic Detection System is an end-to-end solution for real-time vehicle and accident detection using computer vision and deep learning. The system combines:

1. **Machine Learning** (YOLO11) for detecting vehicles and classifying traffic incidents
2. **Backend API** (FastAPI) for event management, persistence, and real-time streaming
3. **Frontend Dashboard** (HTML/JS) for monitoring and visualization
4. **Geolocation mapping** to track where the camera is deployed

The architecture is built for **local deployment** with a focus on:
- Real-time detection and alerts
- Persistent storage of events
- Live dashboard updates via Server-Sent Events
- Configurable camera location tracking
- User authentication and session management

---

## Technology Stack & Roles

### Backend Infrastructure

**FastAPI + Uvicorn**
- **Role**: Serves as the core REST API server for all frontend requests and model output ingestion
- **Key responsibilities**:
  - Accept detection events from `vehicle_detector.py`
  - Expose SSE endpoint for real-time frontend updates
  - Provide authentication (JWT tokens) and session management
  - Return detection history and summary statistics
- **Why chosen**: FastAPI is lightweight, async-capable, and provides automatic Swagger documentation

**Python 3**
- **Role**: Primary language for backend business logic and orchestration
- **Key responsibilities**:
  - Database operations and queries
  - Authentication and token validation
  - Event processing and streaming logic

### Data Persistence

**SQLite**
- **Role**: Lightweight local database for storing all detection events and user data
- **Key responsibilities**:
  - Store detection records (status, vehicle count, confidence, timestamp)
  - Store user credentials and authentication tokens
  - Enable historical queries and reporting
- **Why chosen**: Zero-configuration, file-based database suitable for local deployment; no external server needed

### Machine Learning & Computer Vision

**Ultralytics YOLO11**
- **Role**: Deep learning backbone for object detection and classification
- **Key responsibilities**:
  - Detect vehicles (cars, motorcycles, buses, trucks) in camera frames
  - Classify scenes as `NORMAL`, `CONGESTION`, or `ACCIDENT` based on vehicle overlap
  - Output bounding box coordinates and confidence scores
- **Why chosen**: State-of-the-art real-time detection; pretrained weights available; easy to fine-tune

**OpenCV**
- **Role**: Computer vision library for frame capture, preprocessing, and visualization
- **Key responsibilities**:
  - Read frames from camera or video source
  - Draw bounding boxes and status overlays
  - Compute metrics (FPS, frame dimensions)
  - Log frames to disk if needed
- **Why chosen**: Industry-standard, widely supported, and efficient for real-time video processing

### Frontend Framework

**HTML/CSS/JavaScript**
- **Role**: User interface and dashboard for monitoring detection status
- **Key responsibilities**:
  - Display live detection status and alerts
  - Render historical charts and trends
  - Show camera location on map
  - Allow user configuration (thresholds, camera settings)
  - Handle user authentication (login/signup)
- **Why chosen**: Standard web stack; no build tools required; easy to customize

**Chart.js**
- **Role**: Client-side charting library for visualizing detection trends
- **Key responsibilities**:
  - Render hourly/daily detection counts
  - Display breakdown by status (NORMAL, CONGESTION, ACCIDENT)
  - Update charts in real-time as new data arrives
- **Why chosen**: Lightweight, responsive, and works well with SSE updates

**Leaflet**
- **Role**: Interactive mapping library for displaying camera location
- **Key responsibilities**:
  - Render OpenStreetMap tiles
  - Place camera marker at configured latitude/longitude
  - Allow panning and zooming
- **Why chosen**: Lightweight, open-source, and doesn't require API keys for OSM

### Real-Time Communication

**Server-Sent Events (SSE)**
- **Role**: Push-based protocol for streaming detection updates to the frontend
- **Key responsibilities**:
  - Maintain persistent connection from browser to backend
  - Push new detection events as they occur
  - Enable live dashboard without polling
- **Why chosen**: Simpler than WebSockets; built into modern browsers; unidirectional (server → client) is sufficient for this use case

### Authentication & Security

**JWT (JSON Web Tokens) via custom `auth.py`**
- **Role**: Stateless authentication mechanism for user sessions
- **Key responsibilities**:
  - Generate tokens on login/signup
  - Validate tokens on API requests
  - Manage user roles (admin, user)
- **Why chosen**: Stateless tokens scale well; no session server needed; tokens can be refreshed client-side

### Browser Storage

**localStorage**
- **Role**: Client-side persistence for user preferences and configuration
- **Key responsibilities**:
  - Save camera latitude/longitude
  - Store detection thresholds and stream settings
  - Remember authentication tokens
- **Why chosen**: Built-in browser API; simple key-value store; survives page refreshes

---

## Technology Roles Summary

| Technology | Role | Why It's Needed |
|---|---|---|
| FastAPI | Backend API server | Handles all HTTP requests, SSE, and event processing |
| SQLite | Database | Persistent storage of detections and user data |
| YOLO11 | ML model | Detects vehicles and classifies traffic incidents |
| OpenCV | Video I/O & processing | Reads camera frames and processes video data |
| JavaScript | Frontend logic | Implements dashboard UI and real-time updates |
| Chart.js | Data visualization | Renders charts for analytics and trends |
| Leaflet | Map visualization | Displays camera location on interactive map |
| SSE | Real-time communication | Pushes detection events to connected browsers |
| JWT | Authentication | Secures API endpoints and manages user sessions |
| localStorage | Client-side storage | Persists user preferences and settings |

---

## Methodology


### 1. Model training

- Use `train_accident_model.py` to prepare and train a custom accident detection model.
- The script fixes dataset paths in `car_accident_dataset/data.yaml` and reuses `yolo11n.pt` as the base model.
- Training results are saved under `accident_training/`.

### 2. Detection flow

- Run `vehicle_detector.py` to load the YOLO11 model and process camera frames.
- The detector extracts vehicle bounding boxes, computes overlap, and determines whether the scene is `NORMAL`, `CONGESTION`, or `ACCIDENT`.
- Detection status, vehicle counts, and confidence are packaged into events.

### 3. Backend ingestion

- Detection events are POSTed to `backend/main.py` at `/api/detections`.
- Each event is stored in SQLite and immediately emitted to connected clients using SSE.
- The backend also supports authentication, session management, and summary endpoints.

### 4. Real-time dashboard

- The frontend connects to `/api/events` and listens for server-pushed updates.
- When a new detection event arrives, the dashboard updates:
  - status badge
  - alert log
  - detection details
  - chart data and metrics
- The frontend also supports manual refresh of camera location and persistent settings.

### 5. Camera mapping

- The live page includes a Leaflet map that shows the camera location.
- Camera latitude/longitude values are configurable in `frontend/settings.html`.
- Updated coordinates persist in browser storage and can be refreshed in the live view.

## Architecture Overview

1. **Data capture**: YOLO11 model processes video or camera frames.
2. **Event generation**: Detection status and metadata are created from model output.
3. **API ingestion**: The backend saves events to SQLite and broadcasts them.
4. **Client updates**: Frontend receives live SSE updates and renders them instantly.
5. **Visualization**: Charts, logs, and a location map present the current system state.

## Notes

- The system is designed for local deployment and rapid prototyping.
- It can be extended with additional camera feeds, alerting, and cloud storage.
- The map and dashboard are separated from the backend by HTTP, making the frontend portable.
