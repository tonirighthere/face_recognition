# Luồng Chi Tiết Hệ Thống — Face Recognition

## 1. Kiến trúc tổng quan

```mermaid
graph TB
    Browser["🌐 Trình duyệt\n(192.168.82.65)"]

    subgraph Docker["Docker Network: face_recognition_default"]
        Nginx["🔀 nginx:alpine\nContainer: face_recognition_nginx\nPort 80 → redirect HTTPS\nPort 443 → TLS termination"]
        FastAPI["⚡ FastAPI (uvicorn)\nContainer: face_recognition_api\nPort 8000 (nội bộ)"]
    end

    subgraph Storage["Volumes (Persistent)"]
        DB["🗄️ SQLite\n./database/faces.db"]
        Photos["🖼️ Ảnh gốc\n./storage/photos/"]
        Weights["🤖 Model weights\n./weights/"]
    end

    Browser -- "HTTPS :443 / WSS" --> Nginx
    Browser -- "HTTP :80" --> Nginx
    Nginx -- "HTTP proxy\nws://face_recognition_api:8000" --> FastAPI
    FastAPI <--> DB
    FastAPI <--> Photos
    FastAPI <--> Weights
```

---

## 2. Khởi động hệ thống (Startup)

```mermaid
sequenceDiagram
    participant Docker
    participant uvicorn
    participant lifespan as FastAPI lifespan
    participant deps as dependencies.py
    participant DB as DatabaseManager
    participant Engine as FaceEngine (Singleton)
    participant Store as VectorStore (Singleton)

    Docker->>uvicorn: docker compose up
    uvicorn->>lifespan: startup event
    lifespan->>deps: startup()

    deps->>DB: DatabaseManager() — init SQLite, tạo bảng nếu chưa có
    deps->>Engine: FaceEngine.instance().load()
    Engine->>Engine: _load_insightface() — load ArcFace ONNX (CPU/CUDA)
    Engine->>Engine: _load_mediapipe() — load blaze_face_short_range.tflite
    deps->>Store: VectorStore.instance().load_from_db(db)
    Store->>DB: SELECT id, ho_ten, embedding FROM persons
    Store->>Store: Nạp toàn bộ embeddings vào RAM (numpy array)

    lifespan-->>uvicorn: ✅ Hệ thống sẵn sàng
    Note over Store: Log: "N khuôn mặt trong bộ nhớ"
```

---

## 3. Luồng Webcam Realtime (WebSocket)

```mermaid
sequenceDiagram
    participant Cam as 📷 Webcam
    participant JS as app.js (Browser)
    participant WS as WebSocket /api/recognize/stream
    participant Engine as FaceEngine
    participant Tracker as SimpleTracker
    participant Store as VectorStore

    JS->>Cam: getUserMedia() — yêu cầu quyền camera
    Cam-->>JS: MediaStream (640×480)
    JS->>WS: new WebSocket(wss://host/api/recognize/stream)
    WS-->>JS: accept()

    loop Mỗi 40ms (~25 FPS gửi lên server)
        JS->>JS: drawImage(video → offscreen canvas)
        JS->>JS: canvas.toBlob() → FileReader → base64 JPEG
        JS->>WS: send JSON { frame: base64, threshold: 0.4 }

        WS->>WS: base64 decode → cv2.imdecode → BGR frame

        alt frame_count % SKIP_FRAME == 1 (Chạy detect & tracking update)
            rect rgba(30,80,200,0.1)
                Note over WS,Engine: Bước 1: Detect + Quality Filter
                WS->>Engine: engine.detect(frame, apply_filter=True)
                Engine->>Engine: BGR→RGB → mediapipe.Image
                Engine->>Engine: FaceDetector.detect()
                Engine->>Engine: is_face_valid() — kiểm tra blur / roll / yaw / pitch
                Engine-->>WS: List[BBox (x1,y1,x2,y2,conf)]
            end

            rect rgba(200,100,30,0.1)
                Note over WS: Bước 2: Lọc diện tích tối thiểu
                WS->>WS: filter bbox có area < MIN_FACE_AREA
            end

            rect rgba(30,150,80,0.1)
                Note over WS,Tracker: Bước 3: Cập nhật Tracker (IoU matching)
                WS->>Tracker: tracker.update(bboxes)
                Tracker->>Tracker: Hungarian algorithm / IoU matching
                Tracker-->>WS: List[Track] (track_id ổn định qua frames)
            end
        else frame_count % SKIP_FRAME != 1 (Skip detection, dùng cache tracker)
            WS->>Tracker: Lấy tracker.active_tracks (dùng bbox cũ)
        end

        rect rgba(150,30,150,0.1)
            Note over WS,Store: Bước 4: Embed + Nhận diện (có cache)
            loop Mỗi track active
                alt is_detect_frame VÀ (track mới HOẶC hết TRACK_COOLDOWN giây)
                    WS->>Engine: get_embedding(frame, track.bbox)
                    Engine->>Engine: Crop + padding 30% → InsightFace.get()
                    Engine->>Engine: Align 5 landmarks → ArcFace embed 512D → L2 normalize
                    Engine-->>WS: embedding [512] float32
                    WS->>Store: store.search_best(emb, threshold)
                    Store->>Store: cosine_similarity(emb, all_embeddings)
                    Store-->>WS: (person_id, name, similarity) hoặc None
                    WS->>Tracker: cập nhật cache vào track
                else dùng cache từ track (không re-embed/search)
                    WS->>WS: dùng track.person_name, track.similarity
                end
                WS->>WS: _draw_track() — vẽ bbox + tên lên frame
            end
        end

        WS->>WS: cv2.imencode JPEG q=75 → base64
        WS-->>JS: JSON { image_base64, detections[], fps }
        JS->>JS: drawImage(annotated frame) lên canvas overlay
        JS->>JS: Cập nhật FPS display + detection log sidebar
    end
```

---

## 4. Luồng Nhận diện từ Ảnh tĩnh (REST)

```mermaid
sequenceDiagram
    participant JS as app.js (Browser)
    participant API as POST /api/recognize/image
    participant Engine as FaceEngine
    participant Store as VectorStore

    JS->>API: FormData { photo: File, threshold: 0.4 }
    API->>API: photo.read() → _decode_image() → BGR frame

    API->>Engine: engine.detect(frame, apply_filter=False)
    Note right of Engine: apply_filter=False vì ảnh tĩnh\ncó thể là selfie chụp nghiêng

    loop Mỗi bbox phát hiện được
        API->>Engine: get_embedding(frame, bbox)
        Engine-->>API: embedding 512D
        API->>Store: store.search_best(emb, threshold)
        Store-->>API: (person_id, name, similarity) hoặc None
        API->>API: vẽ bbox + label màu xanh/đỏ
    end

    API->>API: cv2.imencode JPEG q=85 → base64
    API-->>JS: JSON { image_base64, detections[], face_count }
    JS->>JS: Hiển thị ảnh kết quả + danh sách nhận diện
```

---

## 5. Luồng Đăng ký Nhân sự (CRUD)

```mermaid
sequenceDiagram
    participant JS as app.js (Browser)
    participant API as POST /api/persons/
    participant Engine as FaceEngine
    participant DB as DatabaseManager
    participant Store as VectorStore

    JS->>API: FormData { ho_ten, cccd, ngay_sinh, gioi_tinh, dien_thoai, photo }

    API->>Engine: get_embedding_from_image(image)
    Engine->>Engine: detect(apply_filter=False) — phải đúng 1 khuôn mặt
    Engine->>Engine: get_embedding() → 512D vector
    Engine-->>API: embedding hoặc None (lỗi nếu != 1 mặt)

    API->>DB: INSERT INTO persons (..., embedding BLOB)
    DB-->>API: person_id

    API->>Store: store.add(person_id, name, embedding)
    Note right of Store: Cập nhật RAM ngay lập tức\nkhông cần restart

    API->>API: Lưu ảnh gốc vào ./storage/photos/{id}.jpg
    API-->>JS: JSON { message, person_id }
    JS->>JS: Toast "Đăng ký thành công" + reload bảng nhân sự
```

---

## 6. Cấu trúc dữ liệu quan trọng

### VectorStore (in-memory)
```
RAM:
  embeddings: np.ndarray  shape=(N, 512)  float32
  ids:        List[int]   — person_id tương ứng
  names:      List[str]   — tên tương ứng

search_best(query_emb, threshold):
  scores = embeddings @ query_emb  (cosine similarity, đã L2-normalize)
  best_idx = argmax(scores)
  if scores[best_idx] >= threshold → trả về (ids[best_idx], names[best_idx], scores[best_idx])
  else → None
```

### Track (Tracker)
```python
Track:
  track_id:        int       # ID ổn định qua frames
  bbox:            BBox      # vị trí hiện tại
  embedding:       ndarray?  # cache embedding 512D
  last_embed_time: float     # timestamp lần embed gần nhất
  person_id:       int?      # kết quả nhận diện cache
  person_name:     str?
  similarity:      float
  recognized:      bool
  lost:            int       # số frame liên tiếp không thấy
```

---

## 7. Tóm tắt các endpoint

| Method | URL | Mô tả |
|--------|-----|--------|
| `GET` | `/health` | Health check, trả về số người trong RAM |
| `GET` | `/` | Serve `static/index.html` |
| `POST` | `/api/recognize/image` | Nhận diện ảnh tĩnh |
| `WS` | `/api/recognize/stream` | Webcam realtime qua WebSocket |
| `POST` | `/api/persons/` | Đăng ký nhân sự mới |
| `GET` | `/api/persons/` | Danh sách tất cả nhân sự |
| `GET` | `/api/persons/search?q=` | Tìm kiếm nhân sự |
| `DELETE` | `/api/persons/{id}` | Xóa nhân sự |
