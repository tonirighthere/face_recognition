import os
from pathlib import Path

# Đường dẫn gốc   
BASE_DIR = Path(__file__).parent.resolve()
STORAGE_DIR = BASE_DIR / "storage" / "photos"
MODEL_DIR   = BASE_DIR / "weights"
DB_PATH     = BASE_DIR / "database" / "face_recognition.db"

# Tạo thư mục nếu chưa có
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Camera
# CAMERA_INDEX   = "rtsp://root:Atin%40123@192.168.82.134/axis-media/media.amp"
CAMERA_INDEX   = 0      # Webcam mặc định
CAMERA_WIDTH   = 640
CAMERA_HEIGHT  = 480
CAMERA_FPS     = 30

#  Pipeline & Hiển thị 
QUEUE_MAXSIZE  = 3          # Kích thước tối đa của các hàng đợi (đảm bảo Real-time)
QUEUE_TIMEOUT  = 0.1        # Thời gian chờ Queue (giây)
THREAD_SLEEP   = 0.05       # Thời gian ngủ khi Thread pause (giây)
DISPLAY_FPS    = 30         # Tốc độ làm mới giao diện (QTimer pull)
UI_RESULT_EMIT_RATE = 0.5   # Tốc độ cập nhật danh sách log nhận diện (giây)
FPS_CALC_INTERVAL   = 1.0   # Tốc độ cập nhật số FPS trên màn hình (giây)

#  Face Detection (MediaPipe Tasks API — chạy thuần CPU, không cần GPU)
MP_MIN_DETECTION_CONFIDENCE = 0.5   # Ngưỡng confidence tối thiểu của MediaPipe
# Model file: weights/blaze_face_short_range.tflite (tải tự động khi lần đầu chạy)

#  Bộ lọc chất lượng khuôn mặt (Face Quality Filter)
FILTER_BLUR_THRESHOLD  = 50.0    # Laplacian variance; thấp hơn ngưỡng → mờ → loại (webcam thường ~40–80)
FILTER_ROLL_THRESHOLD  = 35.0    # Độ nghiêng đầu trái/phải (°); nới lỏng từ 20 -> 35
FILTER_YAW_MIN         = 0.4     # Tỷ lệ dist(nose,left_eye)/dist(nose,right_eye) tối thiểu (cho phép quay mặt nhiều hơn)
FILTER_YAW_MAX         = 1.6     # Tỷ lệ trên tối đa (ngoài khoảng → quay mặt → loại)
FILTER_PITCH_THRESHOLD = 1.5     # Độ lệch mũi so với đường mắt (đã normalize theo eye_dist) - nới lỏng để cho phép cúi/ngẩng sâu hơn

#  Face Embedding (InsightFace ArcFace) 
INSIGHTFACE_MODEL  = "buffalo_sc"
INSIGHTFACE_DEVICE = -1                  # -1=CPU (Do InsightFace chỉ crop mặt rất nhỏ nên chạy CPU siêu nhanh, tránh lỗi thiếu CUDA rời)

#  Vector Search 
SIMILARITY_THRESHOLD = 0.4             # Ngưỡng nhận diện (cosine similarity)
TOP_K                = 1               # Số kết quả trả về

#  Tracking & Re-embed Logic 
TRACK_MAX_LOST       = 30               # Frame tối đa giữ track khi mất
TRACK_RETAIN_SECONDS = 1.0              # (Giây) Thời gian tối đa giữ bbox trên màn hình khi tracker lag
TRACK_IOU_THRESHOLD  = 0.35             # Ngưỡng IoU để ghép khuôn mặt cũ - mới
TRACK_COOLDOWN       = 1.0              # (Giây) Thời gian chờ giữa 2 lần lấy embedding của cùng 1 người
TRACK_DELTA_CONF     = 0.1              # Tỉ lệ tự tin tăng thêm để lấy lại embedding mới
TRACK_AREA_RATIO     = 1.2              # Tỉ lệ diện tích tăng thêm để lấy lại embedding mới
TRACK_EMA_ALPHA      = 0.8              # Hệ số trung bình động (EMA) khi gộp vector cũ/mới

#  UI 
APP_NAME    = "Face Recognition System"
APP_VERSION = "1.0.0"
WINDOW_W    = 1400
WINDOW_H    = 860

# Màu sắc (Light theme theo mẫu)
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

# Bounding Box Colors (OpenCV BGR Format)
BBOX_COLOR_KNOWN   = (0, 230, 100)
BBOX_COLOR_UNKNOWN = (80, 80, 255)
