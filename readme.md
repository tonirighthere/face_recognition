# 🚀 Face Recognition Desktop App

Dự án Hệ thống Nhận diện Khuôn mặt (Face Recognition Desktop App) là một tổ hợp chặt chẽ giữa Lập trình Giao diện (GUI), Trí tuệ Nhân tạo (Computer Vision / AI), và Quản trị Cơ sở dữ liệu (Database). Hệ thống hỗ trợ nhận diện khuôn mặt realtime từ camera và quản lý thông tin nhân sự.

## 🎯 Tính năng chính
- **Nhận diện khuôn mặt realtime** từ luồng webcam (Liveview).
- **Quản lý thông tin người dùng** (Thêm, Sửa, Xóa, Tìm kiếm danh sách nhân sự).
- **Hiệu năng cao** với thiết kế xử lý đa luồng, loại bỏ hoàn toàn tình trạng giật lag giao diện (non-blocking UI).

## 🏗️ Kiến trúc hệ thống
Hệ thống được thiết kế theo mô hình đa luồng (Multi-threading) tách biệt giữa giao diện và xử lý AI:
- **UI (PyQt5)**: Luồng chính (Main Thread) tập trung hiển thị giao diện và nhận sự kiện từ người dùng.
- **AI Worker (QThread)**: Luồng phụ liên tục đọc frame từ webcam bằng `cv2.VideoCapture()`, xử lý khung hình qua các mô hình AI và đo FPS.
- **Giao tiếp Signals/Slots**: Đẩy kết quả khung hình (đã vẽ bounding box) an toàn từ luồng AI lên luồng giao diện mà không gây crash.
- **Database & Cache**: Sử dụng SQLite lưu trữ thông tin/đặc trưng, kết hợp đọc dữ liệu thành RAM Cache dưới dạng ma trận NumPy khi khởi động để tối ưu tốc độ so khớp.

## 🔴 Pipeline AI & Computer Vision
Pipeline xử lý AI được tối ưu theo quy trình tuần tự 4 bước:
1. **Phát hiện khuôn mặt (YOLOv8-face)**: Dùng mô hình `yolov8n-face.pt` nhỏ nhẹ, chạy trên GPU (CUDA) để phát hiện cả những khuôn mặt nhỏ và ở xa.
2. **Theo dõi đối tượng (Custom IoU Tracker)**: Thuật toán dựa trên IoU do chính dự án tự xây dựng để cấp ID tạm thời cho mỗi khuôn mặt, giúp bỏ qua việc nhận diện lại ở các khung hình liên tiếp và tiết kiệm tài nguyên.
3. **Trích xuất đặc trưng (InsightFace - MobileFaceNet)**: Xoay thẳng mặt và xuất embedding vector (512 chiều). Đặc biệt, hệ thống dùng kỹ thuật *Crop Padding* (pad 30%) và điều hướng chạy trên CPU để tránh lỗi sụp đổ bộ nhớ VRAM.
4. **So khớp Vector (Cosine Similarity)**: Tính toán khoảng cách vector qua NumPy, nếu độ tương đồng lớn hơn ngưỡng cài đặt (ví dụ: `0.38`), xác định thành công người quen.

## 🧠 Công nghệ sử dụng (Tech Stack)
- **Frontend / GUI**: PyQt5
- **Computer Vision**: OpenCV
- **Face Detection**: YOLOv8-face
- **Face Tracking**: Custom IoU Tracker (Thuật toán tự xây dựng)
- **Face Recognition**: InsightFace (`buffalo_sc`)
- **Toán học & Ma trận**: NumPy / FAISS
- **Database**: SQLite

## 🧩 Cấu trúc thư mục dự án

```text
face_recognition/
├── controllers/    # Controller điều khiển luồng nghiệp vụ
├── core/           # Xử lý lõi AI (YOLO, Tracker, InsightFace)
├── database/       # Logic tương tác SQLite
├── models/         # Khai báo các đối tượng/thực thể
├── storage/        # Lưu trữ hình ảnh người dùng (ảnh crop/ảnh upload)
├── utils/          # Các hàm hỗ trợ dùng chung
├── views/          # Mã nguồn giao diện UI/UX
├── weights/        # Chứa file mô hình AI đã huấn luyện
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
   *Lưu ý: Thư viện `torch` sẽ tự động được tải kèm `ultralytics`. Tuy nhiên, để chạy YOLOv8 bằng GPU, bạn cần tải đúng phiên bản PyTorch CUDA từ [pytorch.org](https://pytorch.org).*

3. **Chạy ứng dụng:**
   ```bash
   python main.py
   ```


## ✅ Lộ trình (Roadmap)
- [x] **Phase 1:** Phát hiện khuôn mặt & Trích xuất Embedding.
- [x] **Phase 2:** So khớp Cosine Similarity.
- [x] **Phase 3:** Xây dựng quản lý DB với SQLite (CRUD).
- [x] **Phase 4:** Giao diện Desktop hoàn chỉnh với PyQt5.
- [x] **Phase 5:** Tối ưu hóa đa luồng, theo dõi đối tượng (Custom IoU Tracker), tối ưu RAM Cache.
