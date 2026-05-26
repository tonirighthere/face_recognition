import time
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from config import *
from controllers.ai_worker import AIWorker
from models.db_manager import DatabaseManager
from utils.qt_utils import cv_to_qpixmap, load_scaled_pixmap

class LiveViewWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.worker = AIWorker()

        # Kết nối trực tiếp vào stream_thread để Qt tự dùng QueuedConnection giao frame
        # lên main thread mà không bị drop
        self.worker.stream_thread.recognition_result.connect(self.update_info)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.status_changed.connect(lambda msg: print(f"[Worker] {msg}"))
        self.is_playing = False

        # QTimer để pull frame từ shared_state (giải quyết signal flooding)
        self._display_timer = QTimer(self)
        self._display_timer.setInterval(1000 // DISPLAY_FPS)
        self._display_timer.timeout.connect(self._pull_frame)
        
        self.setup_ui()
        
    @pyqtSlot(str)
    def show_error(self, err_msg):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Lỗi Camera", err_msg)
        self.stop_camera()
        
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(20)
        
        # ─── BÊN TRÁI: KHUNG CAMERA ───────────────────────────────────────────
        left_panel = QFrame()
        left_panel.setStyleSheet(f"""
            QFrame {{
                background-color: #1e212b;
                border-radius: 10px;
                border: 2px solid {COLOR_BORDER};
            }}
        """)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Nút LIVE / START
        top_cam_layout = QHBoxLayout()
        self.lbl_live_status = QLabel("⚫ OFFLINE")
        self.lbl_live_status.setStyleSheet("color: #ef4444; font-weight: bold; background: transparent; border: none;")
        self.lbl_live_status.setFont(QFont("Segoe UI", 12))
        
        self.btn_toggle_cam = QPushButton(" Bật Camera ")
        self.btn_toggle_cam.setFixedSize(120, 35)
        self.btn_toggle_cam.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_cam.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_BLUE};
                color: white;
                border-radius: 5px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{ background-color: #2563eb; }}
        """)
        self.btn_toggle_cam.clicked.connect(self.toggle_camera)
        
        top_cam_layout.addWidget(self.lbl_live_status)
        top_cam_layout.addStretch()
        top_cam_layout.addWidget(self.btn_toggle_cam)
        left_layout.addLayout(top_cam_layout)
        
        # Label hiển thị hình ảnh
        self.lbl_camera = QLabel()
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_camera.setStyleSheet("background: transparent; border: none;")
        left_layout.addWidget(self.lbl_camera, 1)
        
        main_layout.addWidget(left_panel, 7) # Chiếm 7 phần
        
        # ─── BÊN PHẢI: THÔNG TIN NGƯỜI NHẬN DIỆN ──────────────────────────────
        right_panel = QFrame()
        right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_BG_CARD};
                border-radius: 10px;
                border: 2px solid {COLOR_BORDER};
            }}
        """)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)
        
        lbl_title = QLabel("Thông tin người nhận diện")
        lbl_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("border: none;")
        right_layout.addWidget(lbl_title)
        
        # Avatar
        self.lbl_avatar = QLabel("Ảnh")
        self.lbl_avatar.setFixedSize(150, 150)
        self.lbl_avatar.setAlignment(Qt.AlignCenter)
        self.lbl_avatar.setStyleSheet(f"""
            QLabel {{
                background-color: #e2e8f0;
                border: 2px solid {COLOR_BORDER};
                border-radius: 15px;
                color: {COLOR_TEXT_MUTED};
                font-size: 16px;
            }}
        """)
        
        avatar_layout = QHBoxLayout()
        avatar_layout.addWidget(self.lbl_avatar)
        right_layout.addLayout(avatar_layout)
        
        # Thông tin chi tiết
        self.info_labels = {}
        fields = [
            ("Họ tên", "ho_ten"),
            ("Ngày sinh", "ngay_sinh"),
            ("CMND/CCCD", "cccd"),
            ("Giới tính", "gioi_tinh"),
            ("Thời gian", "time"),
            ("Độ tin cậy", "conf")
        ]
        
        info_font = QFont("Segoe UI", 12)
        for label_text, key in fields:
            lbl = QLabel(f"{label_text}: -")
            lbl.setFont(info_font)
            lbl.setStyleSheet("border: none; color: #374151;")
            self.info_labels[key] = lbl
            right_layout.addWidget(lbl)
            
        right_layout.addStretch()
        
        # Nút trạng thái
        self.btn_status = QPushButton("Trạng thái: Chưa nhận diện")
        self.btn_status.setFixedHeight(45)
        self.btn_status.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.btn_status.setStyleSheet(f"""
            QPushButton {{
                background-color: #f1f5f9;
                color: {COLOR_TEXT_MUTED};
                border: 2px solid {COLOR_BORDER};
                border-radius: 10px;
            }}
        """)
        right_layout.addWidget(self.btn_status)
        
        main_layout.addWidget(right_panel, 3) # Chiếm 3 phần
        
    def toggle_camera(self):
        if not self.is_playing:
            self.worker.start_stream()
            self._display_timer.start()
            self.btn_toggle_cam.setText(" Tắt Camera ")
            self.btn_toggle_cam.setStyleSheet(f"QPushButton {{ background-color: {COLOR_BTN_RED}; color: white; border-radius: 5px; font-weight: bold; border: none; }}")
            self.lbl_live_status.setText("🔴 LIVE")
            self.is_playing = True
        else:
            self._display_timer.stop()
            self.worker.stop_stream()
            self.btn_toggle_cam.setText(" Bật Camera ")
            self.btn_toggle_cam.setStyleSheet(f"QPushButton {{ background-color: {COLOR_BTN_BLUE}; color: white; border-radius: 5px; font-weight: bold; border: none; }}")
            self.lbl_live_status.setText("⚫ OFFLINE")
            self.lbl_camera.clear()
            self.clear_info()
            self.is_playing = False

    def stop_camera(self):
        if self.is_playing:
            self.toggle_camera()

    def _pull_frame(self):
        """QTimer callback — chạy ở main thread, chủ động kéo frame mới nhất."""
        frame = None
        with self.worker.shared_state["lock"]:
            frame = self.worker.shared_state.get("display_frame")
        if frame is None:
            return
        pix = cv_to_qpixmap(frame)
        self.lbl_camera.setPixmap(
            pix.scaled(self.lbl_camera.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        )

    @pyqtSlot(object)
    def update_info(self, results):
        if not results:
            return
            
        # Ưu tiên lấy người đã được nhận diện với độ tin cậy cao nhất, nếu không có thì lấy người to nhất
        recognized = [r for r in results if r["recognized"]]
        if recognized:
            best = max(recognized, key=lambda x: x["similarity"])
            person_id = best["person_id"]
            
            # Lấy thông tin từ DB
            db_person = self.db.get_by_id(person_id)
            if db_person:
                self.info_labels["ho_ten"].setText(f"Họ tên: {db_person.get('ho_ten', '')}")
                self.info_labels["ngay_sinh"].setText(f"Ngày sinh: {db_person.get('ngay_sinh', '')}")
                self.info_labels["cccd"].setText(f"CMND/CCCD: {db_person.get('cccd', '')}")
                self.info_labels["gioi_tinh"].setText(f"Giới tính: {db_person.get('gioi_tinh', '')}")
                
                import datetime
                now = datetime.datetime.now().strftime("%H:%M:%S %d/%m")
                self.info_labels["time"].setText(f"Thời gian: {now}")
                
                sim_pct = round(best["similarity"] * 100)
                self.info_labels["conf"].setText(f"Độ tin cậy: {sim_pct}%")
                
                # Cập nhật Avatar nếu có
                if db_person.get('anh_path'):
                    pix = load_scaled_pixmap(db_person['anh_path'], 150, 150, keep_aspect=False)
                    if not pix.isNull():
                        self.lbl_avatar.setPixmap(pix)
                else:
                    self.lbl_avatar.setText("Không có ảnh")
                    
                # Nút trạng thái
                self.btn_status.setText("Trạng thái: Đã nhận diện")
                self.btn_status.setStyleSheet(f"""
                    QPushButton {{
                        background-color: #dcfce7;
                        color: #166534;
                        border: 2px solid {COLOR_BTN_GREEN};
                        border-radius: 10px;
                    }}
                """)
        else:
            # Chỉ có Unknown
            self.clear_info()
            self.btn_status.setText("Trạng thái: Đang theo dõi...")
            self.btn_status.setStyleSheet(f"""
                QPushButton {{
                    background-color: #fef9c3;
                    color: #854d0e;
                    border: 2px solid #eab308;
                    border-radius: 10px;
                }}
            """)
            
    def clear_info(self):
        self.info_labels["ho_ten"].setText("Họ tên: -")
        self.info_labels["ngay_sinh"].setText("Ngày sinh: -")
        self.info_labels["cccd"].setText("CMND/CCCD: -")
        self.info_labels["gioi_tinh"].setText("Giới tính: -")
        self.info_labels["time"].setText("Thời gian: -")
        self.info_labels["conf"].setText("Độ tin cậy: -")
        self.lbl_avatar.clear()
        self.lbl_avatar.setText("Ảnh")
        
        self.btn_status.setText("Trạng thái: Chưa nhận diện")
        self.btn_status.setStyleSheet(f"""
            QPushButton {{
                background-color: #f1f5f9;
                color: {COLOR_TEXT_MUTED};
                border: 2px solid {COLOR_BORDER};
                border-radius: 10px;
            }}
        """)
