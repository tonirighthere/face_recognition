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
