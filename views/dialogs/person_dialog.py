"""
app/dialogs/person_dialog.py — Dialog Thêm / Sửa thông tin người
"""

import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton,
    QFileDialog, QMessageBox, QFrame, QDateEdit,
    QProgressBar,
)
from PyQt5.QtCore import QDate

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import STORAGE_DIR, COLOR_BG_PANEL, COLOR_BG_CARD, COLOR_ACCENT, COLOR_ACCENT2
from config import COLOR_SUCCESS, COLOR_DANGER, COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_BORDER, COLOR_BG_DARK
from core.face_engine import FaceEngine
from utils.qt_utils import load_scaled_pixmap


DIALOG_STYLE = f"""
    QDialog {{
        background: {COLOR_BG_DARK};
        color: {COLOR_TEXT};
    }}
    QLabel {{
        color: {COLOR_TEXT};
        font-size: 13px;
    }}
    QLineEdit, QComboBox, QDateEdit {{
        background: {COLOR_BG_PANEL};
        color: {COLOR_TEXT};
        border: 1px solid {COLOR_BORDER};
        border-radius: 6px;
        padding: 6px 10px;
        font-size: 13px;
        min-height: 32px;
    }}
    QLineEdit:focus, QComboBox:focus, QDateEdit:focus {{
        border-color: {COLOR_ACCENT};
    }}
    QComboBox::drop-down {{ border: none; }}
    QComboBox QAbstractItemView {{
        background: {COLOR_BG_PANEL};
        color: {COLOR_TEXT};
        selection-background-color: {COLOR_ACCENT};
    }}
"""


class EmbedWorker(QThread):
    """Thread riêng để tạo embedding khỏi block UI."""
    finished = pyqtSignal(object, str)   # (embedding|None, message)

    def __init__(self, image_path: str):
        super().__init__()
        self._path = image_path

    def run(self):
        engine = FaceEngine.instance()
        if not engine.is_loaded:
            try:
                engine.load()
            except Exception as e:
                self.finished.emit(None, f"Lỗi load model: {e}")
                return
        img = cv2.imread(self._path)
        if img is None:
            self.finished.emit(None, "Không đọc được ảnh.")
            return
        n = engine.count_faces(img)
        if n == 0:
            self.finished.emit(None, "Không phát hiện khuôn mặt trong ảnh.")
            return
        if n > 1:
            self.finished.emit(None, f"Phát hiện {n} khuôn mặt — chỉ được có đúng 1 người.")
            return
        emb = engine.get_embedding_from_image(img)
        if emb is None:
            self.finished.emit(None, "Không thể tạo embedding. Thử ảnh khác.")
            return
        self.finished.emit(emb, "OK")


class PersonDialog(QDialog):
    """Dialog Thêm / Sửa người dùng trong DB."""

    def __init__(self, person_data: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self._data   = person_data or {}
        self._is_edit = bool(person_data)
        self._embedding: Optional[np.ndarray] = None
        self._image_path: str = ""
        self._embed_worker: Optional[EmbedWorker] = None

        title = "Sửa thông tin" if self._is_edit else "Thêm người mới"
        self.setWindowTitle(title)
        self.setMinimumSize(560, 620)
        self.setStyleSheet(DIALOG_STYLE)
        self._setup_ui()
        if self._is_edit:
            self._fill_data()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(18)

        # Title
        title_lbl = QLabel("👤 " + ("Sửa thông tin" if self._is_edit else "Thêm người mới"))
        title_lbl.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 18px; font-weight: 700;")
        root.addWidget(title_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {COLOR_BORDER};")
        root.addWidget(sep)

        # Body: form + avatar side-by-side
        body = QHBoxLayout()
        body.setSpacing(24)

        # ── Form ──────────────────────────────────────────────────────────
        form_widget = QFrame()
        form_widget.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_BG_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 10px;
                padding: 4px;
            }}
        """)
        form = QFormLayout(form_widget)
        form.setSpacing(12)
        form.setContentsMargins(16, 16, 16, 16)
        form.setLabelAlignment(Qt.AlignRight)

        self._inp_name  = QLineEdit()
        self._inp_name.setPlaceholderText("Họ và tên...")
        self._inp_cccd  = QLineEdit()
        self._inp_cccd.setPlaceholderText("Số CCCD...")
        self._inp_phone = QLineEdit()
        self._inp_phone.setPlaceholderText("Số điện thoại...")

        self._date_edit = QDateEdit()
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setDate(QDate(2000, 1, 1))
        self._date_edit.setDisplayFormat("dd/MM/yyyy")

        self._cmb_gender = QComboBox()
        self._cmb_gender.addItems(["Nam", "Nữ", "Khác"])

        form.addRow("Họ tên *", self._inp_name)
        form.addRow("CCCD", self._inp_cccd)
        form.addRow("Ngày sinh", self._date_edit)
        form.addRow("Giới tính", self._cmb_gender)
        form.addRow("Điện thoại", self._inp_phone)
        body.addWidget(form_widget, 1)

        # ── Avatar ────────────────────────────────────────────────────────
        avatar_col = QVBoxLayout()
        avatar_col.setSpacing(10)
        avatar_col.setAlignment(Qt.AlignTop)

        self._avatar = QLabel()
        self._avatar.setFixedSize(160, 160)
        self._avatar.setAlignment(Qt.AlignCenter)
        self._avatar.setStyleSheet(f"""
            QLabel {{
                background: {COLOR_BG_PANEL};
                border: 2px dashed {COLOR_BORDER};
                border-radius: 10px;
                color: {COLOR_TEXT_MUTED};
                font-size: 12px;
            }}
        """)
        self._avatar.setText("Chưa chọn ảnh")

        self._btn_upload = QPushButton("📁  Chọn ảnh")
        self._btn_upload.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_ACCENT2}; color: #fff;
                border: none; border-radius: 7px;
                padding: 7px 14px; font-size: 12px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #0891b2; }}
        """)

        self._lbl_face_status = QLabel("")
        self._lbl_face_status.setWordWrap(True)
        self._lbl_face_status.setAlignment(Qt.AlignCenter)
        self._lbl_face_status.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED};")

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setFixedHeight(4)
        self._progress.setVisible(False)
        self._progress.setStyleSheet(f"""
            QProgressBar {{ background: {COLOR_BG_PANEL}; border: none; border-radius: 2px; }}
            QProgressBar::chunk {{ background: {COLOR_ACCENT}; border-radius: 2px; }}
        """)

        avatar_col.addWidget(self._avatar)
        avatar_col.addWidget(self._btn_upload)
        avatar_col.addWidget(self._progress)
        avatar_col.addWidget(self._lbl_face_status)
        avatar_col.addStretch()
        body.addLayout(avatar_col)

        root.addLayout(body)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self._btn_cancel = QPushButton("Huỷ")
        self._btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_BG_PANEL}; color: {COLOR_TEXT};
                border: 1px solid {COLOR_BORDER}; border-radius: 8px;
                padding: 9px 24px; font-size: 13px;
            }}
            QPushButton:hover {{ border-color: {COLOR_ACCENT}; }}
        """)
        self._btn_cancel.setFixedHeight(40)

        self._btn_save = QPushButton("💾  Lưu")
        self._btn_save.setStyleSheet(f"""
            QPushButton {{
                background: {COLOR_ACCENT}; color: #fff;
                border: none; border-radius: 8px;
                padding: 9px 28px; font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ background: #6d28d9; }}
            QPushButton:disabled {{ background: #2d2d4e; color: #555; }}
        """)
        self._btn_save.setFixedHeight(40)

        btn_row.addWidget(self._btn_cancel)
        btn_row.addWidget(self._btn_save)
        root.addLayout(btn_row)

        # ── Connect ───────────────────────────────────────────────────────
        self._btn_upload.clicked.connect(self._on_upload)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_cancel.clicked.connect(self.reject)

    # ── Fill data (edit mode) ─────────────────────────────────────────────────

    def _fill_data(self):
        self._inp_name.setText(self._data.get("ho_ten", ""))
        self._inp_cccd.setText(self._data.get("cccd", ""))
        self._inp_phone.setText(self._data.get("dien_thoai", ""))
        gender = self._data.get("gioi_tinh", "Nam")
        idx = self._cmb_gender.findText(gender)
        if idx >= 0:
            self._cmb_gender.setCurrentIndex(idx)
        ngay_sinh = self._data.get("ngay_sinh", "")
        if ngay_sinh:
            try:
                parts = ngay_sinh.split("-")
                self._date_edit.setDate(QDate(int(parts[0]), int(parts[1]), int(parts[2])))
            except Exception:
                pass
        anh_path = self._data.get("anh_path", "")
        if anh_path and Path(anh_path).exists():
            self._set_avatar(anh_path)
            self._image_path = anh_path
        # Khi sửa, cho phép lưu ngay cả khi không re-upload ảnh
        self._lbl_face_status.setText("✅ Dữ liệu cũ đang được dùng" if anh_path else "")

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh đại diện", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        self._set_avatar(path)
        self._image_path = path
        self._lbl_face_status.setText("Đang phân tích khuôn mặt…")
        self._progress.setVisible(True)
        self._btn_save.setEnabled(False)
        self._embedding = None

        self._embed_worker = EmbedWorker(path)
        self._embed_worker.finished.connect(self._on_embed_done)
        self._embed_worker.start()

    def _on_embed_done(self, emb, msg: str):
        self._progress.setVisible(False)
        if emb is None:
            self._lbl_face_status.setStyleSheet(f"font-size: 11px; color: {COLOR_DANGER};")
            self._lbl_face_status.setText(f"❌  {msg}")
            self._btn_save.setEnabled(False)
        else:
            self._embedding = emb
            self._lbl_face_status.setStyleSheet(f"font-size: 11px; color: {COLOR_SUCCESS};")
            self._lbl_face_status.setText("✅  Nhận diện khuôn mặt thành công!")
            self._btn_save.setEnabled(True)

    def _set_avatar(self, path: str):
        pix = load_scaled_pixmap(path, 150, 150, keep_aspect=False)
        self._avatar.setPixmap(pix)
        self._avatar.setAlignment(Qt.AlignCenter)

    def _on_save(self):
        name = self._inp_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Thiếu thông tin", "Vui lòng nhập họ tên.")
            return
        if not self._is_edit and self._embedding is None:
            QMessageBox.warning(self, "Thiếu ảnh", "Vui lòng chọn ảnh chứa khuôn mặt.")
            return
        self.accept()

    # ── Public getters ────────────────────────────────────────────────────────

    def get_data(self) -> dict:
        # Copy ảnh vào STORAGE_DIR
        saved_path = self._data.get("anh_path", "")
        if self._image_path and Path(self._image_path).exists():
            ext  = Path(self._image_path).suffix
            dest_name = f"{self._inp_cccd.text().strip() or self._inp_name.text().strip().replace(' ','_')}{ext}"
            dest = STORAGE_DIR / dest_name
            shutil.copy2(self._image_path, dest)
            saved_path = str(dest)

        qdate = self._date_edit.date()
        ngay_sinh = f"{qdate.year():04d}-{qdate.month():02d}-{qdate.day():02d}"

        return {
            "ho_ten":     self._inp_name.text().strip(),
            "cccd":       self._inp_cccd.text().strip(),
            "ngay_sinh":  ngay_sinh,
            "gioi_tinh":  self._cmb_gender.currentText(),
            "dien_thoai": self._inp_phone.text().strip(),
            "anh_path":   saved_path,
            "embedding":  self._embedding,
        }
