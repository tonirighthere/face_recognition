# Kế hoạch tái cấu trúc: API + Docker + Web UI Realtime

## Tổng quan thay đổi

| Thành phần | Trước | Sau |
|---|---|---|
| Backend | Gradio + PyQt5 | **FastAPI + Uvicorn** |
| Giao diện | Gradio UI / PyQt5 Desktop | **Web UI thuần HTML/CSS/JS** |
| Webcam | PyQt5 QCamera thread | **WebSocket stream (base64 JPEG)** |
| Port | 7860 (Gradio) | **8000 (FastAPI)** |
| Dependencies | PyQt5, gradio | Loại bỏ hoàn toàn |

---

## Kiến trúc mới

```
face_recognition/
├── main.py                    ← FastAPI app (startup, CORS, routes, static)
├── config.py                  ← Đã dọn dẹp (bỏ Qt/Gradio config)
├── requirements.txt           ← fastapi + uvicorn + opencv-headless
├── Dockerfile                 ← Port 8000, uvicorn CMD
├── docker-compose.yml         ← healthcheck /health
│
├── api/
│   ├── dependencies.py        ← Singleton init (startup hook)
│   └── routers/
│       ├── persons.py         ← CRUD: POST/GET/DELETE /api/persons/
│       └── recognize.py       ← POST /api/recognize/image + WS /stream
│
├── core/                      ← Giữ nguyên (FaceEngine, Tracker)
├── models/                    ← Giữ nguyên (DatabaseManager, VectorStore)
├── utils/                     ← Bỏ qt_utils.py, giữ face/tracking/image
│
└── static/                    ← Web UI mới
    ├── index.html             ← 4 tab: Webcam / Ảnh / Nhân sự / API Docs
    ├── style.css              ← Dark theme modern
    └── app.js                 ← WebSocket stream + CRUD logic
```

---

## Các file đã xóa (legacy)

- `views/` — toàn bộ PyQt5 UI
- `controllers/` — threads (camera, detect, face, track, stream)
- `app_gradio.py` — Gradio app
- `docker_setup_guide.md` — guide cũ
- `utils/qt_utils.py` — Qt helpers

---

## API Endpoints

### REST
| Method | URL | Chức năng |
|--------|-----|-----------|
| GET | `/health` | Health check + số nhân sự |
| POST | `/api/persons/` | Đăng ký nhân sự (multipart: info + ảnh) |
| GET | `/api/persons/` | Danh sách |
| GET | `/api/persons/search?q=` | Tìm kiếm |
| GET | `/api/persons/{id}` | Chi tiết |
| GET | `/api/persons/{id}/photo` | Ảnh chân dung |
| DELETE | `/api/persons/{id}` | Xóa |
| POST | `/api/recognize/image` | Nhận diện ảnh tĩnh |

### WebSocket
```
WS /api/recognize/stream

Client → { "frame": "<base64 JPEG>", "threshold": 0.4, "apply_filter": false }
Server → { "image_base64": "...", "detections": [...], "fps": 12.5 }
```

---

## Web UI Features

- **Tab Webcam**: Camera thật → canvas overlay → WebSocket → kết quả realtime + log sidebar
- **Tab Ảnh**: Drag & drop / upload → nhận diện → bounding box trên ảnh kết quả
- **Tab Nhân sự**: Form đăng ký + ảnh upload, bảng danh sách, tìm kiếm, xóa với modal confirm
- **Tab API Docs**: Link Swagger UI / ReDoc + bảng endpoint reference

---

## Chạy nhanh

```bash
# Development
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker compose up -d --build

# Truy cập
# http://localhost:8000        → Web UI
# http://localhost:8000/docs   → Swagger
```

> [!IMPORTANT]
> **Lần đầu chạy** cần tải model MediaPipe:
> ```bash
> python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite', 'weights/blaze_face_short_range.tflite')"
> ```

> [!TIP]
> InsightFace (`buffalo_sc`) tự động tải về `weights/` khi startup lần đầu — không cần thao tác thêm.
