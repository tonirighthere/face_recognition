import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStackedWidget
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon

from config import *
from app.views.liveview_widget import LiveViewWidget
from app.views.crud_widget import CrudWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(WINDOW_W, WINDOW_H)
        
        # Style chung cho toàn bộ app
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLOR_BG_MAIN};
            }}
            QWidget {{
                color: {COLOR_TEXT};
                font-family: 'Segoe UI', Arial, sans-serif;
            }}
            QLabel {{
                color: {COLOR_TEXT};
            }}
        """)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)
        
        # 1. Header Bar
        self.header_bar = QLabel("Camera AI — Nhận diện khuôn mặt (Liveview)")
        self.header_bar.setAlignment(Qt.AlignCenter)
        self.header_bar.setFixedHeight(50)
        font = QFont("Segoe UI", 16)
        self.header_bar.setFont(font)
        self.main_layout.addWidget(self.header_bar)
        
        # Container cho phần còn lại (có padding)
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(20, 0, 20, 20)
        content_layout.setSpacing(15)
        self.main_layout.addWidget(content_container, 1)
        
        # 2. Tabs (Nút bấm)
        tab_layout = QHBoxLayout()
        tab_layout.setSpacing(15)
        
        self.btn_tab_liveview = QPushButton("Liveview")
        self.btn_tab_liveview.setFixedSize(180, 45)
        self.btn_tab_liveview.setCursor(Qt.PointingHandCursor)
        self.btn_tab_liveview.clicked.connect(lambda: self.switch_tab(0))
        
        self.btn_tab_crud = QPushButton("Quản lý người")
        self.btn_tab_crud.setFixedSize(180, 45)
        self.btn_tab_crud.setCursor(Qt.PointingHandCursor)
        self.btn_tab_crud.clicked.connect(lambda: self.switch_tab(1))
        
        tab_layout.addWidget(self.btn_tab_liveview)
        tab_layout.addWidget(self.btn_tab_crud)
        tab_layout.addStretch()
        content_layout.addLayout(tab_layout)
        
        # 3. Stacked Widget (Nội dung chính)
        self.stacked_widget = QStackedWidget()
        self.liveview = LiveViewWidget()
        self.crud = CrudWidget()
        
        self.stacked_widget.addWidget(self.liveview)
        self.stacked_widget.addWidget(self.crud)
        content_layout.addWidget(self.stacked_widget, 1)
        
        # Chọn tab mặc định
        self.switch_tab(0)
        
    def switch_tab(self, index: int):
        self.stacked_widget.setCurrentIndex(index)
        
        # Cập nhật style của header và nút tab
        if index == 0:
            self.header_bar.setText("Camera AI — Nhận diện khuôn mặt (Liveview)")
            self.header_bar.setStyleSheet(f"""
                background-color: {COLOR_ACCENT_BLUE};
                color: {COLOR_TEXT};
                border-bottom: 1px solid {COLOR_BORDER};
            """)
            
            self.btn_tab_liveview.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLOR_ACCENT_BLUE};
                    border: 2px solid {COLOR_BTN_BLUE};
                    border-radius: 20px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)
            self.btn_tab_crud.setStyleSheet(f"""
                QPushButton {{
                    background-color: white;
                    border: 2px solid {COLOR_BORDER};
                    border-radius: 20px;
                    font-size: 14px;
                }}
                QPushButton:hover {{ background-color: #f1f5f9; }}
            """)
        else:
            self.header_bar.setText("Camera AI — Quản lý người (CRUD)")
            self.header_bar.setStyleSheet(f"""
                background-color: {COLOR_ACCENT_PURP};
                color: {COLOR_TEXT};
                border-bottom: 1px solid {COLOR_BORDER};
            """)
            
            self.btn_tab_liveview.setStyleSheet(f"""
                QPushButton {{
                    background-color: white;
                    border: 2px solid {COLOR_BORDER};
                    border-radius: 20px;
                    font-size: 14px;
                }}
                QPushButton:hover {{ background-color: #f1f5f9; }}
            """)
            self.btn_tab_crud.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLOR_ACCENT_PURP};
                    border: 2px solid #8b5cf6;
                    border-radius: 20px;
                    font-size: 14px;
                    font-weight: bold;
                }}
            """)

    def closeEvent(self, event):
        self.liveview.stop_camera()
        event.accept()
