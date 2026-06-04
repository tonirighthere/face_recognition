# 🚀 Face Recognition Desktop App

Dự án Hệ thống Nhận diện Khuôn mặt (Face Recognition Desktop App) là một tổ hợp chặt chẽ giữa Lập trình Giao diện (GUI), Trí tuệ Nhân tạo (Computer Vision / AI), và Quản trị Cơ sở dữ liệu (Database). Hệ thống hỗ trợ nhận diện khuôn mặt realtime từ camera và quản lý thông tin nhân sự.

## 🎯 Tính năng chính
- **Nhận diện khuôn mặt realtime** từ luồng webcam (Liveview).
- **Quản lý thông tin người dùng** (Thêm, Sửa, Xóa, Tìm kiếm danh sách nhân sự).
- **Hiệu năng cao** với thiết kế xử lý đa luồng Pipeline (Multi-Stage Pipeline), loại bỏ hoàn toàn tình trạng giật lag giao diện (non-blocking UI) ngay cả khi chạy thuần CPU.

## 🏗️ Kiến trúc hệ thống
Hệ thống được thiết kế theo mô hình đa luồng (Multi-threading) chia thành một Pipeline tuần tự giúp phân tải xử lý:
- **UI (PyQt5)**: Luồng chính (Main Thread) tập trung hiển thị giao diện và nhận sự kiện từ người dùng.
- **Pipeline AI Workers (QThread)**: Hệ thống gồm các luồng phụ chạy nối tiếp nhau qua các Queue (Hàng đợi):
  - *Camera Thread*: Liên tục đọc frame từ webcam.
  - *Detect Thread*: Phát hiện khuôn mặt bằng MediaPipe.
  - *Track Thread*: Theo dõi đối tượng (Tracking) và cấp ID.
  - *Face Thread*: Trích xuất đặc trưng và nhận diện.
  - *Stream Thread*: Tổng hợp kết quả và đẩy lên giao diện.
- **Giao tiếp Signals/Slots**: Đẩy kết quả khung hình và dữ liệu an toàn từ luồng Pipeline lên luồng giao diện mà không gây crash.
- **Database & Cache**: Sử dụng SQLite lưu trữ thông tin, kết hợp đọc vector đặc trưng thành RAM Cache dưới dạng ma trận NumPy để tối ưu tốc độ so khớp.

## 🔴 Pipeline AI & Computer Vision
Pipeline xử lý AI được tối ưu cho tốc độ và độ chính xác:
1. **Phát hiện khuôn mặt (MediaPipe Tasks API)**: Sử dụng mô hình `BlazeFace` siêu nhẹ, chạy mượt mà trên CPU mà không cần GPU rời.
2. **Bộ lọc chất lượng (Face Quality Filter)**: Đánh giá độ mờ (Blur) và góc nghiêng của khuôn mặt (Yaw, Pitch, Roll) để loại bỏ các khung hình xấu trước khi đưa vào nhận diện.
3. **Theo dõi đối tượng (Custom IoU Tracker)**: Thuật toán dựa trên IoU do chính dự án tự xây dựng để cấp ID tạm thời cho mỗi khuôn mặt, giúp tiết kiệm tài nguyên bằng cách không cần nhận diện lại liên tục ở mọi khung hình.
4. **Trích xuất đặc trưng (InsightFace - MobileFaceNet)**: Xoay thẳng mặt và xuất embedding vector (512 chiều) bằng model `buffalo_sc`.
5. **So khớp Vector (Cosine Similarity)**: Tính toán khoảng cách vector qua NumPy, nếu độ tương đồng lớn hơn ngưỡng cài đặt, hệ thống xác định thành công danh tính người dùng.

## 🧠 Công nghệ sử dụng (Tech Stack)
- **Frontend / GUI**: PyQt5
- **Computer Vision**: OpenCV
- **Face Detection**: MediaPipe (BlazeFace)
- **Face Tracking**: Custom IoU Tracker
- **Face Recognition**: InsightFace (`buffalo_sc`)
- **Toán học & Ma trận**: NumPy
- **Database**: SQLite

## 🧩 Cấu trúc thư mục dự án

```text
face_recognition/
├── controllers/    # Điều khiển nghiệp vụ
│   ├── threads/    # Các luồng AI Pipeline (Camera, Detect, Track, Face, Stream)
│   └── main_controller.py
├── core/           # Xử lý lõi AI (FaceEngine, Tracker)
├── database/       # Chứa file SQLite và script khởi tạo (.sql)
├── models/         # Khai báo các đối tượng thao tác Database & Vector
├── storage/        # Lưu trữ hình ảnh người dùng (ảnh crop/ảnh upload)
├── utils/          # Các hàm hỗ trợ dùng chung (image_utils, qt_utils...)
├── views/          # Mã nguồn giao diện UI/UX
├── weights/        # Chứa file mô hình AI đã huấn luyện (BlazeFace, InsightFace)
├── config.py       # File cấu hình tham số chung toàn hệ thống
├── main.py         # File khởi chạy ứng dụng
└── requirements.txt
```

## ⚙️ Hướng dẫn cài đặt & Chạy dự án

1. **Cài đặt môi trường ảo (Khuyến nghị):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Cài đặt thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Chạy ứng dụng:**
   ```bash
   python main.py
   ```

## ✅ Lộ trình (Roadmap)
- [x] **Phase 1:** Phát hiện khuôn mặt bằng MediaPipe & Trích xuất Embedding InsightFace.
- [x] **Phase 2:** So khớp Cosine Similarity qua **Vector Cache**.
- [x] **Phase 3:** Xây dựng quản lý DB với SQLite (CRUD).
- [x] **Phase 4:** Giao diện Desktop hoàn chỉnh với PyQt5.
- [x] **Phase 5:** Xây dựng kiến trúc Multi-Stage Pipeline (Các Thread nối tiếp), Custom IoU Tracker, và Face Quality Filter.
