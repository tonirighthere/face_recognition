import sys
import logging
import warnings

# Bỏ qua các cảnh báo không mong muốn từ thư viện bên thứ 3 (như insightface/scikit-image)
warnings.filterwarnings("ignore", category=FutureWarning)

import onnxruntime  # QUAN TRỌNG: Phải import onnxruntime trước PyQt5 để tránh lỗi DLL Hell
from pathlib import Path

# Thêm thư mục gốc vào sys.path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from PyQt5.QtWidgets import QApplication, QSplashScreen, QLabel
from PyQt5.QtGui import QFont, QPixmap, QColor, QPainter, QLinearGradient, QPen
from PyQt5.QtCore import Qt, QTimer, QRect, QSize

from config import APP_NAME, APP_VERSION, COLOR_ACCENT_BLUE, COLOR_ACCENT_PURP

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _make_splash(app: QApplication) -> QSplashScreen:
    """Tạo splash screen đẹp."""
    W, H = 520, 300
    pix = QPixmap(W, H)
    pix.fill(Qt.transparent)

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.Antialiasing)

    # Background gradient
    grad = QLinearGradient(0, 0, W, H)
    grad.setColorAt(0.0, QColor("#ffffff"))
    grad.setColorAt(1.0, QColor("#f1f5f9"))
    painter.setBrush(grad)
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(0, 0, W, H, 18, 18)

    # Border
    painter.setPen(QPen(QColor(COLOR_ACCENT_BLUE), 2.0))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(1, 1, W - 2, H - 2, 17, 17)

    # Icon
    painter.setFont(QFont("Segoe UI Emoji", 46))
    painter.setPen(QColor("#1e293b"))
    painter.drawText(QRect(0, 40, W, 80), Qt.AlignCenter, "🧠")

    # Title
    painter.setFont(QFont("Segoe UI", 22, QFont.Bold))
    painter.setPen(QColor("#1e293b"))
    painter.drawText(QRect(0, 130, W, 40), Qt.AlignCenter, APP_NAME)

    # Subtitle
    painter.setFont(QFont("Segoe UI", 11))
    painter.setPen(QColor("#64748b"))
    painter.drawText(QRect(0, 172, W, 28), Qt.AlignCenter, f"v{APP_VERSION}  •  YOLOv8 + InsightFace + PyQt5")

    # Loading
    painter.setFont(QFont("Segoe UI", 10))
    painter.setPen(QColor("#8b5cf6"))
    painter.drawText(QRect(0, 250, W, 30), Qt.AlignCenter, "Đang khởi động…")

    painter.end()

    splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
    splash.setFont(QFont("Segoe UI", 10))
    return splash


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Font mặc định
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Splash screen
    splash = _make_splash(app)
    splash.show()
    app.processEvents()

    # Import lazy để splash hiện trước
    from views.main_window import MainWindow
    window = MainWindow()

    # Ẩn splash sau 1.5 giây, hiện main window
    QTimer.singleShot(1500, splash.close)
    QTimer.singleShot(1500, window.show)

    logger.info(f"{APP_NAME} v{APP_VERSION} started.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
