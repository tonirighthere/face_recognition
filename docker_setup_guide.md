# 🐳 Hướng Dẫn Cài Docker & Deploy Gradio Web UI cho Dự Án Face Recognition
> **Máy chủ:** `atin@192.168.82.65` (SSH nội bộ công ty)  
> **Dự án:** Face Recognition System v1.0.0 — Gradio Web Interface + MediaPipe + InsightFace + ONNX

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
9. [Truy cập giao diện Demo & Lưu ý bảo mật Webcam (HTTPS)](#9-truy-cập-giao-diện-demo--lưu-ý-bảo-mật-webcam-https)
10. [Expose link Internet để show demo cho khách hàng từ xa](#10-expose-link-internet-để-show-demo-cho-khách-hàng-từ-xa)
11. [Quản lý container](#11-quản-lý-container)
12. [Troubleshooting](#12-troubleshooting)

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

> Hướng dẫn này dành cho **Ubuntu 20.04 / 22.04** (phổ biến nhất).

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

*   **Nếu dùng Git Bash / MSYS2:**
    ```bash
    rsync -avz --exclude='venv/' --exclude='__pycache__/' \
        /d/intern/Atin/face_recognition/ \
        atin@192.168.82.65:/home/atin/face_recognition/
    ```

*   **Nếu dùng WSL (Windows Subsystem for Linux):**
    ```bash
    rsync -avz --exclude='venv/' --exclude='__pycache__/' \
        /mnt/d/intern/Atin/face_recognition/ \
        atin@192.168.82.65:/home/atin/face_recognition/
    ```

---

## 6. Tạo Dockerfile

Tạo file `Dockerfile` trong thư mục gốc dự án:

```bash
nano /home/atin/face_recognition/Dockerfile
```

Nội dung `Dockerfile` (Đã tối ưu hóa cho Gradio, lược bỏ hoàn toàn PyQt5 X11/VNC GUI dependencies cồng kềnh giúp giảm dung lượng image):

```dockerfile
# ============================================================
# Base image: Python 3.10 slim trên Debian Bullseye
# ============================================================
FROM python:3.10-slim-bullseye

LABEL maintainer="atin@company.com"
LABEL description="Face Recognition System - InsightFace + MediaPipe + Gradio Web UI"

# ============================================================
# 1. Cài các thư viện hệ thống cần thiết (chủ yếu cho OpenCV)
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
    # Utilities
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 2. Thiết lập thư mục làm việc
# ============================================================
WORKDIR /app

# ============================================================
# 3. Cài Python dependencies
# ============================================================
COPY requirements.txt .

# Nâng pip và cài thư viện
RUN pip install --upgrade pip && \
    pip install --no-cache-dir onnxruntime>=1.16.0 && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# 4. Copy mã nguồn dự án
# ============================================================
COPY . .

# ============================================================
# 5. Tạo các thư mục dữ liệu
# ============================================================
RUN mkdir -p storage/photos database weights

# ============================================================
# 6. Biến môi trường
# ============================================================
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose cổng Gradio (7860)
EXPOSE 7860

# Khởi chạy ứng dụng Gradio
CMD ["python", "app_gradio.py"]
```

---

## 7. Tạo docker-compose.yml

```bash
nano /home/atin/face_recognition/docker-compose.yml
```

Nội dung `docker-compose.yml`:

```yaml
version: "3.9"

services:
  face_recognition_web:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: face_recognition_gradio
    restart: unless-stopped

    environment:
      - PYTHONUNBUFFERED=1

    volumes:
      # Persistent data: database SQLite
      - ./database:/app/database
      # Persistent data: ảnh nhân sự
      - ./storage:/app/storage
      # Persistent data: model weights
      - ./weights:/app/weights

    ports:
      # Gradio Web port — dùng trình duyệt để xem demo
      - "7860:7860"

    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## 8. Build và chạy container

```bash
cd /home/atin/face_recognition

# Build image (mất khoảng 5-10 phút trong lần build đầu tiên)
docker compose build

# Chạy container ở chế độ chạy ngầm (detached)
docker compose up -d

# Xem log kiểm tra khởi động
docker compose logs -f face_recognition_web
```

Khi thấy dòng chữ `Running on local URL: http://0.0.0.0:7860` xuất hiện trong log, hệ thống đã sẵn sàng hoạt động.

---

## 9. Truy cập giao diện Demo & Lưu ý bảo mật Webcam (HTTPS)

### Cách truy cập:
*   Mở trình duyệt web của bạn, truy cập vào địa chỉ: `http://192.168.82.65:7860`

> [!IMPORTANT]
> **Lưu ý Quan Trọng Về Quyền Truy Cập Webcam từ xa (HTTPS):**
> *   Các trình duyệt web hiện đại (Chrome, Edge, Safari...) có cơ chế bảo mật nghiêm ngặt: **chỉ cho phép trang web gọi camera (webcam) khi trang web đó sử dụng giao thức bảo mật HTTPS** hoặc chạy trên **localhost / 127.0.0.1**.
> *   Nếu bạn truy cập thông qua link HTTP địa chỉ IP (ví dụ: `http://192.168.82.65:7860`), nút **Webcam** trong tab Nhận diện hoặc Đăng ký sẽ bị trình duyệt **chặn không cho truy cập**.
> *   **Giải pháp:**
>     1. **Sử dụng tính năng "Upload Ảnh"** để demo (tính năng này chạy bình thường ở mọi giao thức).
>     2. **Cấu hình HTTPS** (sử dụng SSL certificate trực tiếp trong Gradio hoặc thông qua Reverse Proxy như Nginx).
>     3. **Sử dụng dịch vụ Tunnel** (Ngrok / Cloudflare Tunnel) để nhận link public tự động có HTTPS bảo mật (Hướng dẫn chi tiết ở mục 10).

---

## 10. Expose link Internet để show demo cho khách hàng từ xa

Để gửi link demo cho khách hàng tự trải nghiệm trên thiết bị/laptop của họ (sử dụng camera của chính họ), bạn cần mở cổng ra internet. Dưới đây là 2 phương pháp miễn phí và bảo mật tốt nhất:

### Cách 1 — Sử dụng Cloudflare Tunnel (Khuyên dùng)
Cloudflare cung cấp đường truyền ổn định, bảo mật và cung cấp sẵn HTTPS miễn phí.

1.  **Cài đặt `cloudflared` trên máy chủ Ubuntu:**
    ```bash
    curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared.deb
    ```
2.  **Khởi tạo Tunnel nhanh:**
    ```bash
    cloudflared tunnel --url http://localhost:7860
    ```
3.  **Lấy link:** Terminal sẽ in ra một đường dẫn ngẫu nhiên có định dạng: `https://xxxx-xxxx-xxxx.trycloudflare.com`.
4.  **Trải nghiệm:** Khách hàng chỉ cần click vào link HTTPS này để dùng trực tiếp Webcam trên thiết bị của họ.

---

### Cách 2 — Sử dụng Ngrok
Dễ thiết lập và rất phổ biến.

1.  Đăng ký tài khoản miễn phí trên [ngrok.com](https://ngrok.com) để lấy token.
2.  **Cài đặt ngrok trên máy chủ:**
    ```bash
    curl -s https://ngrok-agent.s3.amazonaws.com/files.bin/key.lnk | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/ngrok.gpg >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com/files.bin/ubuntu_noarch/ /" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt-get update && sudo apt-get install ngrok
    ```
3.  **Cấu hình Token (Chỉ làm 1 lần):**
    ```bash
    ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>
    ```
4.  **Chạy ngrok expose port 7860:**
    ```bash
    ngrok http 7860
    ```
5.  **Lấy link:** Lấy link dạng `https://xxxx.ngrok-free.app` hiển thị trên màn hình terminal gửi cho khách hàng.

---

## 11. Quản lý container

```bash
# Xem danh sách container đang chạy
docker ps

# Dừng ứng dụng
docker compose stop

# Khởi động lại
docker compose restart

# Xem log realtime
docker compose logs -f face_recognition_web

# Vào trong container để kiểm tra file / debug
docker compose exec face_recognition_web bash

# Xóa container (vẫn giữ nguyên ảnh và database nhờ volume mount)
docker compose down

# Dọn dẹp cache Docker khi cần thiết
docker system prune -f
```

---

## 12. Troubleshooting

### ❌ Lỗi camera trên trình duyệt bị tắt / xám đen
*   **Nguyên nhân:** Chưa dùng giao thức HTTPS. Trình duyệt tự chặn quyền.
*   **Khắc phục:** Sử dụng Cloudflare Tunnel hoặc Ngrok (Mục 10) để truy cập qua link HTTPS.

### ❌ Lỗi `libGL.so.1: cannot open shared object file`
*   **Nguyên nhân:** Thiếu thư viện đồ họa OpenGL trên Linux trong container.
*   **Khắc phục:** Dockerfile đã cài sẵn `libgl1`. Nếu tự chạy bên ngoài container, hãy chạy lệnh: `sudo apt-get install -y libgl1-mesa-glx`.

### ❌ Lỗi InsightFace không tải được model `buffalo_sc`
*   **Nguyên nhân:** Server không có kết nối internet hoặc bị chặn firewall khi tải model lần đầu từ GitHub.
*   **Khắc phục:** Tải model thủ công và chép vào thư mục `weights/models/buffalo_sc/` trên server.

### ❌ Lỗi SQLite bị khóa file database (`database is locked`)
*   **Nguyên nhân:** Có nhiều tiến trình ghi đồng thời vào file `.db`.
*   **Khắc phục:** Khởi động lại container bằng lệnh `docker compose restart`. Giao diện Gradio được thiết kế bất tuần tự (async) nên sẽ hạn chế tối đa xung đột ghi file.
