"""
config.py — Cấu hình toàn hệ thống Face Recognition Desktop App
"""

import os
from pathlib import Path

# ─── Đường dẫn gốc ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = BASE_DIR / "storage" / "photos"
MODEL_DIR   = BASE_DIR / "weights"
DB_PATH     = BASE_DIR / "database" / "face_recognition.db"

# Tạo thư mục nếu chưa có
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─── Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0          # Webcam mặc định
CAMERA_WIDTH   = 1280
CAMERA_HEIGHT  = 720
FRAME_SKIP     = 3          # Chỉ chạy AI mỗi N frame

# ─── Face Detection (YOLOv8-face) ───────────────────────────────────────────
YOLO_MODEL_NAME    = "yolov8n-face.pt"   # nano model — nhanh, nhẹ
YOLO_CONF_THRESH   = 0.50
YOLO_IOU_THRESH    = 0.45
YOLO_IMG_SIZE      = 640
YOLO_DEVICE        = "cuda"              # "cuda" nếu có GPU

# ─── Face Embedding (InsightFace ArcFace) ───────────────────────────────────
INSIGHTFACE_MODEL  = "buffalo_sc"        # buffalo_sc=nhỏ/nhanh, buffalo_l=chính xác hơn
INSIGHTFACE_DEVICE = -1                  # -1=CPU (Do InsightFace chỉ crop mặt rất nhỏ nên chạy CPU siêu nhanh, tránh lỗi thiếu CUDA rời)

# ─── Vector Search ──────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.38             # Ngưỡng nhận diện (cosine similarity)
TOP_K                = 1               # Số kết quả trả về

# ─── ByteTrack ──────────────────────────────────────────────────────────────
TRACK_HIGH_THRESH  = 0.6
TRACK_LOW_THRESH   = 0.1
TRACK_MAX_LOST     = 30               # Frame tối đa giữ track khi mất

# ─── UI ─────────────────────────────────────────────────────────────────────
APP_NAME    = "Face Recognition System"
APP_VERSION = "1.0.0"
WINDOW_W    = 1400
WINDOW_H    = 860

# ─── Màu sắc (Light theme theo mẫu) ─────────────────────────────────────────
COLOR_BG_MAIN   = "#ffffff"
COLOR_BG_PANEL  = "#f8f9fa"
COLOR_BG_CARD   = "#ffffff"
COLOR_ACCENT_BLUE = "#93c5fd"   # Màu xanh nhạt của tab Liveview
COLOR_ACCENT_PURP = "#c4b5fd"   # Màu tím nhạt của tab Quản lý
COLOR_BTN_BLUE  = "#3b82f6"     # Nút Sửa
COLOR_BTN_RED   = "#ef4444"     # Nút Xóa
COLOR_BTN_GREEN = "#22c55e"     # Nút Thêm / Lưu
COLOR_TEXT      = "#1f2937"
COLOR_TEXT_MUTED= "#6b7280"
COLOR_BORDER    = "#cbd5e1"
