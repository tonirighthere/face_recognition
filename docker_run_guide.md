# Hướng dẫn chạy Docker Compose

## Yêu cầu cài đặt

| Môi trường | Cần cài |
|---|---|
| Local (Windows) | [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Server (Linux) | `docker` + `docker compose` plugin |

---

## 🖥️ A. Chạy ở LOCAL (Windows)

### Bước 1 — Tải model MediaPipe (chỉ lần đầu)

Mở **PowerShell** trong thư mục `face_recognition/`:

```powershell
# Tạo thư mục weights nếu chưa có
New-Item -ItemType Directory -Force -Path weights

# Tải model MediaPipe
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" `
  -OutFile "weights\blaze_face_short_range.tflite"
```

> **InsightFace** (`buffalo_sc`) tự tải về `weights/` khi container khởi động lần đầu — không cần làm thêm gì.

---

### Bước 2 — Build & Start

```powershell
# Vào thư mục dự án
cd d:\intern\Atin\face_recognition

# Build image và chạy (chạy nền)
docker compose up -d --build
```

Lần đầu build mất **5–15 phút** (tải base image + cài packages).

---

### Bước 3 — Kiểm tra

```powershell
# Xem container đang chạy
docker compose ps

# Xem logs realtime
docker compose logs -f

# Kiểm tra health
curl http://localhost:8000/health
```

**Kết quả health check mong đợi:**
```json
{"status": "ok", "app": "Face Recognition System", "version": "2.0.0", "persons_in_memory": 0}
```

---

### Truy cập Web UI

| URL | Mô tả |
|-----|-------|
| `http://localhost:8000` | 🎯 Web UI demo |
| `http://localhost:8000/docs` | 📖 Swagger API |
| `http://localhost:8000/redoc` | 📄 ReDoc |

---

### Các lệnh thường dùng (Local)

```powershell
# Dừng container (giữ data)
docker compose stop

# Dừng và xóa container (giữ volumes/data)
docker compose down

# Restart
docker compose restart

# Rebuild khi thay đổi code
docker compose up -d --build

# Xem log của lần chạy gần nhất
docker compose logs --tail=100
```

---

## 🌐 B. Chạy ở SERVER (Linux — 192.168.82.65)

### Bước 1 — Copy code lên server

Chạy từ **Git Bash / WSL** trên máy Windows:

```bash
# Dùng rsync (khuyến nghị)
rsync -avz --progress \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.git/' \
  /d/intern/Atin/face_recognition/ \
  atin@192.168.82.65:/home/atin/face_recognition/
```

> Hoặc nếu đã dùng **Git**: push code → pull trên server.

---

### Bước 2 — SSH vào server

```bash
ssh atin@192.168.82.65
cd /home/atin/face_recognition
```

---

### Bước 3 — Cài Docker (nếu server chưa có)

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Kiểm tra
docker --version
docker compose version
```

---

### Bước 4 — Tải model MediaPipe (lần đầu)

```bash
mkdir -p weights
curl -L \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" \
  -o weights/blaze_face_short_range.tflite

echo "✅ Model downloaded: $(ls -lh weights/blaze_face_short_range.tflite)"
```

---

### Bước 5 — Build & Start

```bash
# Build image và chạy nền
docker compose up -d --build
```

---

### Bước 6 — Kiểm tra trên server

```bash
# Trạng thái container
docker compose ps

# Logs (theo dõi realtime)
docker compose logs -f

# Health check
curl http://localhost:8000/health

# Xem InsightFace có tải model chưa (lần đầu mất vài phút)
docker compose logs face_recognition | grep -i "insightface\|buffalo\|loaded\|sẵn sàng"
```

---

### Truy cập từ máy khác trong mạng LAN

```
http://192.168.82.65:8000        → Web UI demo
http://192.168.82.65:8000/docs   → Swagger API
```

> [!NOTE]
> Nếu server có **firewall**, mở port 8000:
> ```bash
> sudo ufw allow 8000/tcp
> sudo ufw reload
> ```

---

### Các lệnh thường dùng (Server)

```bash
# Dừng
docker compose stop

# Dừng và xóa container
docker compose down

# Xem 200 dòng log cuối
docker compose logs --tail=200

# Restart sau khi update code
docker compose up -d --build

# Xem tài nguyên đang dùng
docker stats face_recognition_api

# Vào trong container để debug
docker exec -it face_recognition_api bash
```

---

## 🔄 Workflow cập nhật code

```bash
# 1. Sửa code trên Windows → rsync lên server
rsync -avz --exclude='venv/' --exclude='__pycache__/' \
  /d/intern/Atin/face_recognition/ \
  atin@192.168.82.65:/home/atin/face_recognition/

# 2. SSH và rebuild
ssh atin@192.168.82.65
cd /home/atin/face_recognition
docker compose up -d --build

# 3. Kiểm tra
docker compose logs -f
```

---

## ⚠️ Lưu ý quan trọng

> [!IMPORTANT]
> Dữ liệu nhân sự (DB + ảnh) được **mount qua volumes** — không mất khi rebuild hay restart container:
> - `./database/` → SQLite
> - `./storage/photos/` → Ảnh nhân sự  
> - `./weights/` → AI model weights

> [!WARNING]
> **Chỉ dùng `--workers 1`** (đã cấu hình trong Dockerfile). FaceEngine và VectorStore là singleton trong RAM — nhiều workers sẽ gây lỗi state không đồng nhất.

> [!TIP]
> Nếu InsightFace chưa tải model sau 3 phút đầu, xem log:
> ```bash
> docker compose logs face_recognition | tail -50
> ```
> Model `buffalo_sc` cần kết nối internet, dung lượng ~85MB.
