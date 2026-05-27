import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

def cv_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """
    Chuyển đổi numpy array (OpenCV BGR/BGRA) sang QPixmap để hiển thị trên PyQt5.
    """
    h, w, ch = frame.shape
    bytes_per_line = ch * w

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qt_img)

def load_scaled_pixmap(image_path: str, width: int, height: int, keep_aspect: bool = True) -> QPixmap:
    """
    Đọc ảnh từ đường dẫn và scale theo kích thước cho trước.
    """
    pix = QPixmap(image_path)
    if pix.isNull():
        return pix
        
    if keep_aspect:
        # Tương đương object-fit: cover -> scale đầy khung hình rồi crop ở giữa
        scaled = pix.scaled(width, height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        crop_x = (scaled.width() - width) // 2
        crop_y = (scaled.height() - height) // 2
        return scaled.copy(crop_x, crop_y, width, height)
    else:
        return pix.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
