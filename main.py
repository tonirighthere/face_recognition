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

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from config import APP_NAME, APP_VERSION

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Font mặc định
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    from views.main_window import MainWindow
    window = MainWindow()
    window.show()

    logger.info(f"{APP_NAME} v{APP_VERSION} started.")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
