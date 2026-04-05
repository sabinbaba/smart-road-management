# Smart Traffic Detection System

This repository contains a full traffic incident detection system with:
- **Backend API** powered by `FastAPI`
- **Frontend dashboard** served from static HTML/JS/CSS files
- **YOLO11 vehicle/accident detection model** integration
- **Live camera location map** support for monitoring camera placement

---

## Prerequisites

- Python 3.10+ recommended
- Git (optional)
- Internet access for downloading model weights and Leaflet assets

---

## Install dependencies

From the project root:

```bash
python -m pip install --upgrade pip
python -m pip install fastapi uvicorn python-multipart pydantic
python -m pip install ultralytics opencv-python numpy
```

If you want to use the provided `requirements.txt` for the model side, install:

```bash
python -m pip install -r requirements.txt
```

> Note: `requirements.txt` includes `ultralytics`, `opencv-python`, and `numpy`.

---

## Run the backend API

The backend is located in `backend/main.py` and exposes the API for detection events, authentication, and SSE updates.

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once running, verify the backend:

- API root: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- SSE events: `http://localhost:8000/api/events`

---

## Serve the frontend

The frontend files are in the `frontend/` folder.

### Option 1: Use Python's local server

From the project root:

```bash
cd frontend
python -m http.server 8080
```

Open in browser:

```text
http://localhost:8080/index.html
```

### Option 2: Open directly in browser

You can also open `frontend/index.html` directly, but some browsers may block SSE or fetch requests when using `file://`. The local server method is recommended.

---

## Configure the camera location map

The live monitor page includes a camera location map.

- Default coordinates are configured in `frontend/live.html`.
- You can change the camera latitude and longitude in `frontend/settings.html`.
- After saving settings, click **Refresh Map** on the live page to update the marker.

---

## Run the detection model

The repository contains a sample detection script and training helper:

- `vehicle_detector.py` — main detection and camera feed integration using YOLO11
- `train_accident_model.py` — custom training script for accident detection

### Run the detector

This script uses `yolo11n.pt` and an attached camera or video source.

```bash
python vehicle_detector.py
```

> If `vehicle_detector.py` is commented out in the repository, enable the entrypoint or adapt the script for your camera feed.

### Train a custom accident model

If you want to retrain on the included dataset:

```bash
python train_accident_model.py
```

This script will:
- fix dataset paths in `car_accident_dataset/data.yaml`
- train a model using `yolo11n.pt`
- save results under `accident_training/`

---

## Notes

- The frontend connects to the backend at `http://localhost:8000` by default.
- If you run the backend on a different host or port, update `frontend/js/shared.js` and any hard-coded camera stream URL in `frontend/live.html`.
- The system uses SQLite for detection storage, initialized automatically on backend startup.

---

## Folder structure

- `backend/` — FastAPI server, authentication, database API
- `frontend/` — dashboard pages, shared JS, styles
- `car_accident_dataset/` — dataset files for training
- `train_accident_model.py` — model training helper
- `vehicle_detector.py` — YOLO-based detection runner
- `yolo11n.pt` — pretrained YOLO11 model backbone

---

## Troubleshooting

- If the frontend cannot connect to the backend, make sure `UVICORN` is running and CORS is allowed.
- If the map is blank, ensure the browser can load Leaflet from the CDN.
- If the detector fails to open the camera, update `WEBCAM_INDEX` or the camera stream URL in `vehicle_detector.py`.
