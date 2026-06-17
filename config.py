import os
from pathlib import Path

# ── Đường dẫn gốc ──────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
STORAGE_DIR = BASE_DIR / "storage" / "photos"
MODEL_DIR   = BASE_DIR / "weights"
DB_PATH     = BASE_DIR / "database" / "face_recognition.db"

# Tạo thư mục nếu chưa có
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Face Detection (MediaPipe) ──────────────────────────────────────────────
MP_MIN_DETECTION_CONFIDENCE = 0.5

# ── Bộ lọc chất lượng khuôn mặt ────────────────────────────────────────────
FILTER_BLUR_THRESHOLD  = 50.0
FILTER_ROLL_THRESHOLD  = 35.0
FILTER_YAW_MIN         = 0.35
FILTER_YAW_MAX         = 1.3
FILTER_PITCH_THRESHOLD = 1.2

# ── Face Embedding (InsightFace ArcFace) ────────────────────────────────────
INSIGHTFACE_MODEL  = "buffalo_sc"
INSIGHTFACE_DEVICE = -1   # -1 = CPU

# ── Vector Search ───────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.4
TOP_K                = 1

# ── Tracking ─────────────────────────────────────────────────────────────────
TRACK_MAX_LOST       = 30
TRACK_IOU_THRESHOLD  = 0.35
TRACK_COOLDOWN       = 1.0        # giây: thời gian chờ tối thiểu giữa 2 lần re-embed
TRACK_DELTA_CONF     = 0.1
TRACK_AREA_RATIO     = 1.2
TRACK_EMA_ALPHA      = 0.8
MIN_FACE_AREA        = 3600       # pixel²: bỏ qua mặt nhỏ hơn ~60×60px

# ── Bounding Box Colors (OpenCV BGR) ────────────────────────────────────────
BBOX_COLOR_KNOWN   = (0, 230, 100)
BBOX_COLOR_UNKNOWN = (80, 80, 255)

# ── App Info ─────────────────────────────────────────────────────────────────
APP_NAME    = "Face Recognition System"
APP_VERSION = "2.0.0"
API_PORT    = int(os.getenv("API_PORT", 8000))
