# 🚀 Face Recognition System — API + Docker + Web UI

Hệ thống nhận diện khuôn mặt realtime sử dụng **MediaPipe** (face detection) + **InsightFace ArcFace** (face embedding), đóng gói Docker, cung cấp **REST API** + **WebSocket stream**, kèm **Web UI** tích hợp sẵn.

---

## 📐 Kiến trúc

```
face_recognition/
├── main.py                    # FastAPI app entry point
├── config.py                  # Cấu hình toàn cục
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── api/                       # FastAPI layer
│   ├── dependencies.py        # Singleton: DB, FaceEngine, VectorStore
│   └── routers/
│       ├── persons.py         # CRUD nhân sự (POST/GET/DELETE)
│       └── recognize.py       # Nhận diện ảnh + WebSocket stream
│
├── core/                      # AI Engine
│   ├── face_engine.py         # MediaPipe detect + InsightFace embed
│   └── tracker.py             # Face tracker (IoU-based)
│
├── models/                    # Data layer
│   ├── db_manager.py          # SQLite CRUD
│   └── vector_store.py        # RAM cosine similarity search
│
├── utils/
│   ├── face_utils.py          # Bộ lọc chất lượng (blur/roll/yaw/pitch)
│   └── tracking_utils.py     # IoU, EMA helpers
│
├── database/
│   └── schema.sql
│
├── storage/photos/            # Ảnh chân dung nhân sự
├── weights/                   # Model weights (auto-download)
│
└── static/                    # Web UI (HTML/CSS/JS)
    ├── index.html
    ├── style.css
    └── app.js
```

---

## ⚡ Chạy nhanh (Development)

### 1. Cài dependencies
```bash
pip install -r requirements.txt
```

### 2. Tải model weights (lần đầu)
```bash
python -c "
import urllib.request
urllib.request.urlretrieve(
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
  'weights/blaze_face_short_range.tflite'
)
print('MediaPipe model downloaded.')
"
```
> InsightFace (`buffalo_sc`) tự động tải xuống khi khởi động lần đầu.

### 3. Chạy server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Truy cập
| URL | Mô tả |
|-----|-------|
| `http://localhost:8000` | Web UI demo |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/redoc` | ReDoc API docs |
| `http://localhost:8000/health` | Health check |

---

## 🐳 Chạy bằng Docker

### Build & start
```bash
docker compose up -d --build
```

### Xem logs
```bash
docker compose logs -f
```

### Dừng
```bash
docker compose down
```

### Dữ liệu persistent
Dữ liệu được mount qua volumes — **không mất khi restart container**:
- `./database/` → SQLite
- `./storage/photos/` → Ảnh nhân sự
- `./weights/` → Model weights

---

## 🌐 REST API Reference

### Nhân sự (Persons)
| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/persons/` | Đăng ký nhân sự (multipart: thông tin + ảnh) |
| `GET` | `/api/persons/` | Danh sách tất cả |
| `GET` | `/api/persons/search?q=...` | Tìm kiếm |
| `GET` | `/api/persons/{id}` | Chi tiết |
| `GET` | `/api/persons/{id}/photo` | Ảnh chân dung |
| `DELETE` | `/api/persons/{id}` | Xóa |

### Nhận diện (Recognize)
| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/api/recognize/image` | Nhận diện từ ảnh tĩnh (multipart) |
| `WS` | `/api/recognize/stream` | WebSocket realtime stream |

#### WebSocket Protocol
```json
// Client → Server
{ "frame": "<base64 JPEG>", "threshold": 0.4, "apply_filter": false }

// Server → Client
{ "image_base64": "...", "detections": [...], "fps": 12.5 }
```

---

## 🖥️ Deploy lên Server

```bash
# 1. Copy code lên server
rsync -avz --exclude='venv/' --exclude='weights/' \
  /path/to/face_recognition/ atin@192.168.82.65:/home/atin/face_recognition/

# 2. SSH vào server
ssh atin@192.168.82.65
cd /home/atin/face_recognition

# 3. Tải model weights (nếu chưa có)
mkdir -p weights
curl -L "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" \
     -o weights/blaze_face_short_range.tflite

# 4. Build & start Docker
docker compose up -d --build

# 5. Kiểm tra
curl http://localhost:8000/health
```

---

## ⚙️ Cấu hình

Chỉnh sửa `config.py` để thay đổi:
- `SIMILARITY_THRESHOLD` — ngưỡng cosine similarity (mặc định `0.4`)
- `INSIGHTFACE_DEVICE` — `-1` = CPU, `0` = GPU CUDA
- `FILTER_*` — các ngưỡng bộ lọc chất lượng khuôn mặt
