import os
import shutil
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLineEdit, QHeaderView, QMessageBox, QFrame, QLabel,
    QComboBox, QFileDialog, QSizePolicy, QDateEdit
)
from PyQt5.QtCore import Qt, QSize, QDate
from PyQt5.QtGui import QIcon, QPixmap, QFont

from config import *
from models.db_manager import DatabaseManager
from core.face_engine import FaceEngine
from utils.qt_utils import load_scaled_pixmap

class CrudWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.engine = FaceEngine.instance()
        
        self.current_edit_id = None
        self.current_img_path = None
        
        self.setup_ui()
        self.load_data()
        
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(20)
        
        # BÊN TRÁI: DANH SÁCH & TÌM KIẾM
        left_panel = QFrame()
        left_panel.setStyleSheet(f"background-color: {COLOR_BG_CARD}; border-radius: 10px; border: 1px solid {COLOR_BORDER};")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # Toolbar
        toolbar = QHBoxLayout()
        self.txt_search = QLineEdit()
        self.txt_search.setPlaceholderText("Tìm theo tên / CMND...")
        self.txt_search.setFixedHeight(40)
        self.txt_search.setStyleSheet(f"border: 2px solid {COLOR_BORDER}; border-radius: 5px; padding-left: 10px; font-size: 14px; background: white;")
        self.txt_search.textChanged.connect(self.load_data)
        
        self.btn_add = QPushButton(" + Thêm người ")
        self.btn_add.setFixedHeight(40)
        self.btn_add.setCursor(Qt.PointingHandCursor)
        self.btn_add.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_GREEN};
                color: white;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                padding: 0 15px;
                border: none;
            }}
            QPushButton:hover {{ background-color: #16a34a; }}
        """)
        self.btn_add.clicked.connect(self.prepare_add)
        
        toolbar.addWidget(self.txt_search)
        toolbar.addWidget(self.btn_add)
        left_layout.addLayout(toolbar)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Ảnh", "Họ tên", "Ngày sinh", "CMND", "Thao tác"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: white;
                border: none;
                gridline-color: {COLOR_BORDER};
            }}
            QHeaderView::section {{
                background-color: #f1f5f9;
                color: {COLOR_TEXT_MUTED};
                font-weight: bold;
                border: none;
                border-bottom: 2px solid {COLOR_BORDER};
                padding: 5px;
            }}
            QTableWidget::item {{
                padding: 5px;
                border-bottom: 1px solid #f1f5f9;
            }}
        """)
        left_layout.addWidget(self.table)
        main_layout.addWidget(left_panel, 7) # Chiếm 7 phần
        
        # ─── BÊN PHẢI: FORM THÊM / SỬA ────────────────────────────────────────
        self.right_panel = QFrame()
        self.right_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_BG_CARD};
                border-radius: 10px;
                border: 2px solid {COLOR_ACCENT_PURP};
            }}
        """)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)
        
        lbl_title = QLabel("Thêm / Sửa người")
        lbl_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("border: none;")
        right_layout.addWidget(lbl_title)
        
        # Avatar picker
        self.btn_avatar = QPushButton("+ Ảnh")
        self.btn_avatar.setFixedSize(120, 120)
        self.btn_avatar.setCursor(Qt.PointingHandCursor)
        self.btn_avatar.setStyleSheet(f"""
            QPushButton {{
                background-color: #e2e8f0;
                border: 2px solid {COLOR_BORDER};
                border-radius: 15px;
                color: {COLOR_TEXT_MUTED};
                font-size: 14px;
            }}
            QPushButton:hover {{ background-color: #cbd5e1; }}
        """)
        self.btn_avatar.clicked.connect(self.choose_image)
        
        avatar_layout = QHBoxLayout()
        avatar_layout.addWidget(self.btn_avatar)
        right_layout.addLayout(avatar_layout)
        
        # Inputs
        self.input_hoten = QLineEdit()
        self.input_hoten.setPlaceholderText("Họ tên")
        
        self.input_ngaysinh = QDateEdit()
        self.input_ngaysinh.setDisplayFormat("dd/MM/yyyy")
        self.input_ngaysinh.setCalendarPopup(True)
        self.input_ngaysinh.setDate(QDate.currentDate())
        
        self.input_cccd = QLineEdit()
        self.input_cccd.setPlaceholderText("CMND/CCCD")
        
        self.cb_gioitinh = QComboBox()
        self.cb_gioitinh.addItems(["Nam", "Nữ", "Khác"])
        
        self.input_sdt = QLineEdit()
        self.input_sdt.setPlaceholderText("Số điện thoại")
        
        inputs = [self.input_hoten, self.input_ngaysinh, self.input_cccd, self.cb_gioitinh, self.input_sdt]
        for inp in inputs:
            inp.setFixedHeight(40)
            inp.setStyleSheet(f"border: 2px solid {COLOR_BORDER}; border-radius: 5px; padding-left: 10px; font-size: 14px; background: white;")
            right_layout.addWidget(inp)
            
        right_layout.addStretch()
        
        # Nút Lưu / Hủy
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Lưu")
        self.btn_save.setFixedHeight(45)
        self.btn_save.setCursor(Qt.PointingHandCursor)
        self.btn_save.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BTN_GREEN};
                color: white;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }}
            QPushButton:hover {{ background-color: #16a34a; }}
        """)
        self.btn_save.clicked.connect(self.save_person)
        
        self.btn_cancel = QPushButton("Hủy")
        self.btn_cancel.setFixedHeight(45)
        self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background-color: #e2e8f0;
                color: {COLOR_TEXT};
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }}
            QPushButton:hover {{ background-color: #cbd5e1; }}
        """)
        self.btn_cancel.clicked.connect(self.clear_form)
        
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(btn_layout)
        
        main_layout.addWidget(self.right_panel, 3) # Chiếm 3 phần
        self.right_panel.setVisible(False) # Ẩn lúc đầu
        self.clear_form() # Reset ban đầu
        
    def choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh khuôn mặt", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            # Dùng cv2 để đọc ảnh -> check khuôn mặt
            img = cv2.imread(file_path)
            if img is None:
                QMessageBox.warning(self, "Lỗi", "Không thể đọc ảnh!")
                return
                
            self.engine.load()
            faces = self.engine.count_faces(img)
            if faces == 0:
                QMessageBox.warning(self, "Lỗi", "Không tìm thấy khuôn mặt nào trong ảnh!")
                return
            if faces > 1:
                QMessageBox.warning(self, "Lỗi", "Ảnh có quá nhiều khuôn mặt, vui lòng chọn ảnh chân dung 1 người!")
                return
                
            self.current_img_path = file_path
            pix = QPixmap(file_path)
            self.btn_avatar.setIcon(QIcon(pix))
            self.btn_avatar.setIconSize(QSize(110, 110))
            self.btn_avatar.setText("")
            
    def prepare_add(self):
        self.clear_form()
        self.right_panel.setVisible(True)
        self.input_hoten.setFocus()
        
    def prepare_edit(self, person):
        self.clear_form()
        self.right_panel.setVisible(True)
        self.current_edit_id = person['id']
        self.input_hoten.setText(person['ho_ten'])
        ns_str = person.get('ngay_sinh', "")
        date = QDate.fromString(ns_str, "yyyy-MM-dd")
        if not date.isValid():
            date = QDate.fromString(ns_str, "dd/MM/yyyy")
        if date.isValid():
            self.input_ngaysinh.setDate(date)
        else:
            self.input_ngaysinh.setDate(QDate.currentDate())
            
        self.input_cccd.setText(person.get('cccd', ""))
        self.cb_gioitinh.setCurrentText(person.get('gioi_tinh', "Nam"))
        self.input_sdt.setText(person.get('dien_thoai', ""))
        
        if person.get('anh_path') and os.path.exists(person['anh_path']):
            self.current_img_path = person['anh_path']
            pix = QPixmap(person['anh_path'])
            self.btn_avatar.setIcon(QIcon(pix))
            self.btn_avatar.setIconSize(QSize(110, 110))
            self.btn_avatar.setText("")
            
    def clear_form(self):
        self.current_edit_id = None
        self.current_img_path = None
        self.input_hoten.clear()
        self.input_ngaysinh.setDate(QDate.currentDate())
        self.input_cccd.clear()
        self.cb_gioitinh.setCurrentIndex(0)
        self.input_sdt.clear()
        self.btn_avatar.setIcon(QIcon())
        self.btn_avatar.setText("+ Ảnh")
        self.right_panel.setVisible(False)
        
    def save_person(self):
        hoten = self.input_hoten.text().strip()
        cccd = self.input_cccd.text().strip()
        
        if not hoten:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập họ tên!")
            return
            
        # Thêm mới: Bắt buộc có ảnh
        if not self.current_edit_id and not self.current_img_path:
            QMessageBox.warning(self, "Lỗi", "Vui lòng chọn ảnh khuôn mặt để đăng ký!")
            return
            
        emb = None
        saved_img_path = None
        
        # Nếu có chọn ảnh mới -> Trích xuất vector
        if self.current_img_path and (not self.current_edit_id or self.current_img_path != self.db.get_by_id(self.current_edit_id).get('anh_path')):
            self.engine.load()
            img = cv2.imread(self.current_img_path)
            emb = self.engine.get_embedding_from_image(img)
            if emb is None:
                QMessageBox.warning(self, "Lỗi", "Không thể trích xuất đặc trưng khuôn mặt!")
                return
                
            # Lưu file ảnh vào thư mục storage
            if not STORAGE_DIR.exists():
                STORAGE_DIR.mkdir(parents=True)
                
            ext = os.path.splitext(self.current_img_path)[1]
            new_filename = f"user_{cccd or hoten}_{os.urandom(4).hex()}{ext}"
            saved_img_path = str(STORAGE_DIR / new_filename)
            shutil.copy2(self.current_img_path, saved_img_path)
            
        try:
            if self.current_edit_id:
                # Cập nhật
                self.db.update_person(
                    self.current_edit_id, hoten, self.input_ngaysinh.date().toString("yyyy-MM-dd"), cccd,
                    self.cb_gioitinh.currentText(), self.input_sdt.text(),
                    saved_img_path, emb
                )
                QMessageBox.information(self, "Thành công", "Đã cập nhật thông tin!")
            else:
                # Thêm mới
                self.db.add_person(
                    hoten, self.input_ngaysinh.date().toString("yyyy-MM-dd"), cccd,
                    self.cb_gioitinh.currentText(), self.input_sdt.text(),
                    saved_img_path, emb
                )
                QMessageBox.information(self, "Thành công", "Đã thêm người mới!")
                
            self.clear_form()
            self.load_data()
            
            # Cập nhật lại bộ nhớ Vector để tab Liveview nhận dạng ngay lập tức
            from models.vector_store import VectorStore
            VectorStore.instance().load_from_db(self.db)
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi CSDL", str(e))
            
    def delete_person(self, person_id: int):
        reply = QMessageBox.question(
            self, 'Xác nhận xóa', 
            'Bạn có chắc chắn muốn xóa nhân sự này?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.db.delete_person(person_id)
            self.load_data()
            
            # Cập nhật lại bộ nhớ Vector
            from models.vector_store import VectorStore
            VectorStore.instance().load_from_db(self.db)

    def load_data(self):
        keyword = self.txt_search.text().strip()
        persons = self.db.search(keyword)
        
        self.table.setRowCount(len(persons))
        for row, p in enumerate(persons):
            # Ảnh
            lbl_img = QLabel()
            if p.get('anh_path') and os.path.exists(p['anh_path']):
                pix = load_scaled_pixmap(p['anh_path'], 50, 50, keep_aspect=False)
                lbl_img.setPixmap(pix)
            else:
                lbl_img.setText("No Image")
            lbl_img.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(row, 0, lbl_img)
            
            # Text
            self.table.setItem(row, 1, QTableWidgetItem(p['ho_ten']))
            self.table.setItem(row, 2, QTableWidgetItem(p.get('ngay_sinh', "")))
            self.table.setItem(row, 3, QTableWidgetItem(p.get('cccd', "")))
            
            # Căn giữa cho text
            for col in range(1, 4):
                self.table.item(row, col).setTextAlignment(Qt.AlignCenter)
            
            # Thao tác (Sửa / Xóa)
            action_widget = QWidget()
            action_layout = QHBoxLayout(action_widget)
            action_layout.setContentsMargins(5, 5, 5, 5)
            action_layout.setSpacing(5)
            
            btn_edit = QPushButton("Sửa")
            btn_edit.setCursor(Qt.PointingHandCursor)
            btn_edit.setStyleSheet(f"background-color: {COLOR_BTN_BLUE}; color: white; border-radius: 4px; padding: 4px 8px; border: none;")
            btn_edit.clicked.connect(lambda checked, person=p: self.prepare_edit(person))
            
            btn_delete = QPushButton("Xóa")
            btn_delete.setCursor(Qt.PointingHandCursor)
            btn_delete.setStyleSheet(f"background-color: {COLOR_BTN_RED}; color: white; border-radius: 4px; padding: 4px 8px; border: none;")
            btn_delete.clicked.connect(lambda checked, pid=p['id']: self.delete_person(pid))
            
            action_layout.addWidget(btn_edit)
            action_layout.addWidget(btn_delete)
            self.table.setCellWidget(row, 4, action_widget)
            
            self.table.setRowHeight(row, 50)
