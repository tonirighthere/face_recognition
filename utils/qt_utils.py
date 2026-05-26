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
    
    if ch == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
    else:
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
        
    aspect_mode = Qt.KeepAspectRatio if keep_aspect else Qt.IgnoreAspectRatio
    return pix.scaled(width, height, aspect_mode, Qt.SmoothTransformation)
