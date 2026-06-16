# 🐳 Hướng Dẫn Cài Docker cho Dự Án Face Recognition
> **Máy chủ:** `atin@192.168.82.65` (SSH nội bộ công ty)  
> **Dự án:** Face Recognition System v1.0.0 — PyQt5 + MediaPipe + InsightFace + ONNX

---

## 📋 Mục Lục
1. [Kết nối SSH vào máy chủ](#1-kết-nối-ssh-vào-máy-chủ)
2. [Kiểm tra hệ điều hành máy chủ](#2-kiểm-tra-hệ-điều-hành-máy-chủ)
3. [Cài đặt Docker Engine](#3-cài-đặt-docker-engine)
4. [Cài đặt Docker Compose](#4-cài-đặt-docker-compose)
5. [Chuẩn bị code trên máy chủ](#5-chuẩn-bị-code-trên-máy-chủ)
6. [Tạo Dockerfile](#6-tạo-dockerfile)
7. [Tạo docker-compose.yml](#7-tạo-docker-composeyml)
8. [Build và chạy container](#8-build-và-chạy-container)
9. [Chạy UI trên server headless (không màn hình)](#9-chạy-ui-trên-server-headless-không-màn-hình)
10. [Quản lý container](#10-quản-lý-container)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Kết nối SSH vào máy chủ

Mở terminal (PowerShell / CMD / Git Bash) trên máy tính của bạn:

```bash
ssh atin@192.168.82.65
```

> Nhập password khi được hỏi. Nếu muốn không phải nhập password mỗi lần, cấu hình SSH key:
> ```bash
> # Trên máy local của bạn
> ssh-keygen -t ed25519 -C "face_recognition"
> ssh-copy-id atin@192.168.82.65
> ```

---

## 2. Kiểm tra hệ điều hành máy chủ

Sau khi SSH vào, chạy lệnh sau để biết distro:

```bash
lsb_release -a
# Hoặc
cat /etc/os-release
```

> Hướng dẫn này dành cho **Ubuntu 20.04 / 22.04** (phổ biến nhất). Nếu server dùng CentOS/RHEL, báo lại để điều chỉnh.

---

## 3. Cài đặt Docker Engine

### Bước 3.1 — Gỡ phiên bản cũ nếu có
```bash
sudo apt-get remove -y docker docker-engine docker.io containerd runc
```

### Bước 3.2 — Cài các gói phụ thuộc
```bash
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

### Bước 3.3 — Thêm GPG key và repository chính thức của Docker
```bash
# Thêm GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Thêm repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### Bước 3.4 — Cài Docker Engine
```bash
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Bước 3.5 — Cho phép chạy Docker không cần `sudo`
```bash
sudo usermod -aG docker $USER
# Sau đó logout và login lại SSH để có hiệu lực:
exit
ssh atin@192.168.82.65
```

### Bước 3.6 — Kiểm tra cài đặt
```bash
docker --version
# Kết quả mong đợi: Docker version 26.x.x, build ...

docker run hello-world
# Kết quả mong đợi: "Hello from Docker!"
```

---

## 4. Cài đặt Docker Compose

Docker Compose V2 đã được cài tự động ở bước 3.4. Kiểm tra:

```bash
docker compose version
# Kết quả: Docker Compose version v2.x.x
```

> Nếu cần `docker-compose` (v1 cũ) riêng biệt:
> ```bash
> sudo apt-get install -y docker-compose
> docker-compose --version
> ```

---

## 5. Chuẩn bị code trên máy chủ

### Cách A — Clone từ Git (khuyến nghị)
```bash
cd /home/atin
git clone <URL_REPO_CUA_BAN> face_recognition
cd face_recognition
```

### Cách B — Upload từ máy local qua SCP
Trên máy Windows của bạn (PowerShell):
```powershell
# Upload toàn bộ thư mục (trừ venv và __pycache__)
scp -r d:\intern\Atin\face_recognition atin@192.168.82.65:/home/atin/
```

Hoặc dùng **rsync** (nhanh hơn, bỏ qua file không cần thiết):
```bash
rsync -avz --exclude='venv/' --exclude='__pycache__/' \
    d:/intern/Atin/face_recognition/ \
    atin@192.168.82.65:/home/atin/face_recognition/
```

---

## 6. Tạo Dockerfile

> Dự án dùng **PyQt5** (cần display), **InsightFace**, **MediaPipe**, **ONNX Runtime**.  
> Dockerfile dưới đây tối ưu cho môi trường CPU-only trên server Linux.

Tạo file `Dockerfile` trong thư mục gốc dự án:

```bash
# SSH vào server
nano /home/atin/face_recognition/Dockerfile
```

Nội dung `Dockerfile`:

```dockerfile
# ============================================================
# Base image: Python 3.10 slim trên Ubuntu 22.04
# ============================================================
FROM python:3.10-slim-bullseye

# Metadata
LABEL maintainer="atin@company.com"
LABEL description="Face Recognition System - InsightFace + MediaPipe + PyQt5"

# ============================================================
# 1. Cài các dependency hệ thống
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    # PyQt5 / Qt5 dependencies (cần cho GUI)
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    qt5-qmake \
    python3-pyqt5 \
    # X11 / Display (headless xvfb hoặc forward display)
    libx11-6 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    xvfb \
    x11vnc \
    # Multimedia
    libgstreamer1.0-0 \
    # Utilities
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. Thiết lập thư mục làm việc
# ============================================================
WORKDIR /app

# ============================================================
# 3. Cài Python dependencies (cache layer riêng cho tốc độ)
# ============================================================
COPY requirements.txt .

# Nâng pip và cài thư viện
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        # ONNX Runtime TRƯỚC để tránh DLL conflict
        onnxruntime>=1.16.0 \
    && pip install --no-cache-dir \
        PyQt5>=5.15.9 \
        opencv-python-headless>=4.8.0 \
        mediapipe>=0.10.0 \
        insightface>=0.7.3 \
        numpy>=1.24.0

# ============================================================
# 4. Copy source code
# ============================================================
COPY . .

# ============================================================
# 5. Tạo thư mục cần thiết
# ============================================================
RUN mkdir -p storage/photos database weights

# ============================================================
# 6. Biến môi trường
# ============================================================
# Headless Qt (không cần màn hình vật lý — dùng Xvfb)
ENV QT_QPA_PLATFORM=xcb
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ============================================================
# 7. Entrypoint script
# ============================================================
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 5900

ENTRYPOINT ["/docker-entrypoint.sh"]
```

---

## 6b. Tạo Entrypoint Script

```bash
nano /home/atin/face_recognition/docker-entrypoint.sh
```

```bash
#!/bin/bash
set -e

# Khởi động Xvfb (màn hình ảo) để PyQt5 có thể render
Xvfb :99 -screen 0 1920x1080x24 &
sleep 1

# (Tuỳ chọn) Khởi động VNC để xem từ xa qua VNC Viewer
x11vnc -display :99 -nopw -forever -rfbport 5900 &

echo "=== Face Recognition System Starting ==="
exec python main.py "$@"
```

---

## 7. Tạo docker-compose.yml

```bash
nano /home/atin/face_recognition/docker-compose.yml
```

```yaml
version: "3.9"

services:
  face_recognition:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: face_recognition_app
    restart: unless-stopped

    environment:
      - DISPLAY=:99
      - QT_QPA_PLATFORM=xcb
      - PYTHONUNBUFFERED=1

    volumes:
      # Persistent data: database SQLite
      - ./database:/app/database
      # Persistent data: ảnh nhân viên
      - ./storage:/app/storage
      # Persistent data: model weights
      - ./weights:/app/weights
      # (Tuỳ chọn) Mount webcam nếu server có camera USB
      # - /dev/video0:/dev/video0

    devices:
      # Cho phép container truy cập webcam
      # Bỏ comment nếu server có camera vật lý
      # - /dev/video0:/dev/video0

    ports:
      # VNC port — dùng VNC Viewer từ máy tính để xem GUI
      - "5900:5900"

    # Cấp quyền truy cập device (cần cho webcam/camera)
    privileged: false
    # group_add:
    #   - video   # Bỏ comment nếu cần truy cập /dev/video*

    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

> [!NOTE]
> Nếu dùng **camera RTSP** (như cấu hình `CAMERA_INDEX` trong `config.py`), không cần mount `/dev/video0` — chỉ cần đảm bảo server có mạng tới địa chỉ RTSP.

---

## 8. Build và chạy container

```bash
cd /home/atin/face_recognition

# Build image (lần đầu mất 5–15 phút do tải MediaPipe, InsightFace)
docker compose build

# Chạy container ở chế độ nền (detached)
docker compose up -d

# Xem log realtime
docker compose logs -f face_recognition
```

---

## 9. Chạy UI trên server headless (không màn hình)

Vì đây là ứng dụng **PyQt5 Desktop**, bạn có 2 cách xem giao diện:

### Cách 1 — VNC Viewer (khuyến nghị)
1. Cài [VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/) trên máy Windows của bạn.
2. Kết nối tới: `192.168.82.65:5900`
3. Giao diện Face Recognition sẽ hiện ra qua màn hình ảo Xvfb.

### Cách 2 — X11 Forwarding qua SSH
Trên máy Windows (dùng **MobaXterm** hoặc **Xming**):
```bash
# SSH với X11 forwarding
ssh -X atin@192.168.82.65

# Sau đó chạy trực tiếp (không dùng Docker)
cd /home/atin/face_recognition
python main.py
```

### Cách 3 — Chạy trong container với X11 forward
```bash
# Trên server, trước khi chạy Docker:
xhost +local:docker

docker compose run --rm \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    face_recognition
```

---

## 10. Quản lý container

```bash
# Xem danh sách container đang chạy
docker ps

# Dừng ứng dụng
docker compose stop

# Khởi động lại
docker compose restart

# Xem log
docker compose logs -f

# Vào trong container để debug
docker compose exec face_recognition bash

# Xóa container (giữ image)
docker compose down

# Xóa container + image (build lại từ đầu)
docker compose down --rmi local

# Xem dung lượng Docker đang dùng
docker system df

# Dọn dẹp cache không dùng
docker system prune -f
```

---

## 11. Troubleshooting

### ❌ Lỗi `Cannot connect to the X server`
```bash
# Kiểm tra Xvfb đã chạy chưa
ps aux | grep Xvfb

# Chạy thủ công
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
python main.py
```

### ❌ Lỗi `libGL.so.1: cannot open shared object file`
```bash
# Trong container, cài thêm
apt-get install -y libgl1-mesa-glx
```

### ❌ Lỗi InsightFace không tải được model
```bash
# Model buffalo_sc cần internet lần đầu
# Kiểm tra proxy/firewall trên server
curl -I https://github.com
# Nếu cần, copy thủ công model từ máy local:
scp -r ~/.insightface/models/buffalo_sc atin@192.168.82.65:/home/atin/.insightface/models/
```

### ❌ Lỗi camera/webcam không nhận
```bash
# Kiểm tra device tồn tại trên server
ls /dev/video*
# Thêm user vào group video
sudo usermod -aG video atin
# Bỏ comment dòng devices trong docker-compose.yml
```

### ❌ Build chậm / hết dung lượng
```bash
# Kiểm tra disk
df -h
# Dọn dẹp Docker
docker system prune -a -f
```

---

## 📁 Cấu trúc cuối cùng trên server

```
/home/atin/face_recognition/
├── Dockerfile                  ← Image definition
├── docker-compose.yml          ← Service orchestration
├── docker-entrypoint.sh        ← Startup script (Xvfb + VNC + app)
├── requirements.txt
├── main.py
├── config.py
├── controllers/
├── core/
├── database/                   ← Bind mount (persistent)
├── models/
├── storage/photos/             ← Bind mount (persistent)
├── utils/
├── views/
└── weights/                    ← Bind mount (persistent)
```

---

> [!IMPORTANT]
> Sau khi cài xong, kiểm tra lại `config.py` trên server — đặc biệt dòng `CAMERA_INDEX`:
> - Nếu dùng **webcam USB** → `CAMERA_INDEX = 0` và mount `/dev/video0`
> - Nếu dùng **camera RTSP** → `CAMERA_INDEX = "rtsp://root:Atin%40123@192.168.82.134/axis-media/media.amp"`

