Dự án **Hệ thống Nhận diện Khuôn mặt (Face Recognition Desktop App)** này là một tổ hợp rất chặt chẽ giữa **Lập trình Giao diện (GUI)**, **Trí tuệ Nhân tạo (Computer Vision / AI)**, và **Quản trị Cơ sở dữ liệu (Database)**. 

Dưới đây là bức tranh tổng quan về toàn bộ các luồng kiến thức, công nghệ và các kỹ thuật xử lý chuyên sâu mà chúng ta đã áp dụng vào dự án này:

---

### 1. Luồng Trí tuệ Nhân tạo (AI & Computer Vision Pipeline)
Luồng xử lý AI là "trái tim" của hệ thống, được thiết kế theo mô hình **Pipeline tuần tự 4 bước** để tối ưu hóa tốc độ:
*   **Phát hiện khuôn mặt (Face Detection - YOLOv8-face):** Thay vì dùng các model truyền thống, chúng ta dùng phiên bản YOLOv8 đã được train riêng cho việc bắt khuôn mặt (`yolov8n-face.pt`). Nó cực kỳ nhẹ, nhạy với mặt nhỏ và được ép chạy trên **GPU (CUDA)** để có tốc độ realtime.
*   **Theo dõi đối tượng (Object Tracking - ByteTrack):** Áp dụng thuật toán ByteTrack (`core/tracker.py`). Thuật toán này giúp cấp cho mỗi khuôn mặt một "ID tạm thời" (Track ID) khi họ di chuyển trong khung hình. **Mục đích:** Tránh việc phải đem khuôn mặt đi nhận diện lại ở *mọi khung hình* (rất tốn tài nguyên). Nếu ID đó đã được nhận diện, hệ thống chỉ việc "nhớ" tên người đó ở các frame tiếp theo.
*   **Trích xuất đặc trưng (Face Embedding - InsightFace):** Dùng bộ model `buffalo_sc` (MobileFaceNet) siêu nhẹ.
    *   **Kỹ thuật Crop Padding:** Vì YOLO cắt mặt quá sát, chúng ta đã áp dụng kỹ thuật mở rộng khung cắt (pad thêm 30%) để InsightFace có không gian tìm được 5 điểm mốc (mắt, mũi, miệng).
    *   Từ 5 điểm này, nó xoay thẳng mặt (Alignment) và xuất ra một vector **512 chiều (512D float32)** đại diện cho khuôn mặt đó. Để tránh lỗi tràn bộ nhớ card hình, ta đã cấu hình cho bước này chạy trên **CPU**.
*   **So khớp Vector (Cosine Similarity Search):** Dùng thư viện `NumPy` tính toán khoảng cách Cosine giữa Vector đang đứng trước camera và tập hợp toàn bộ Vector trong cơ sở dữ liệu. Nếu độ giống nhau (Similarity) lớn hơn ngưỡng `0.38`, hệ thống kết luận đó là người quen.

### 2. Kiến trúc Hệ thống & Đa luồng (System & Multithreading)
Nếu nhét cả Camera, AI và Giao diện vào cùng một chỗ, app sẽ bị "đơ" ngay lập tức. Chúng ta đã dùng kỹ thuật **Đa luồng (Multi-threading)** của PyQt5:
*   **Main Thread (Luồng chính):** Chỉ chịu trách nhiệm vẽ giao diện (UI), nhận nút bấm, đổi màu sắc. Tuyệt đối không làm toán hay xử lý AI ở đây.
*   **Worker Thread (`AIWorker`):** Kế thừa `QThread`. Luồng này chạy ngầm bên dưới, liên tục đọc ảnh từ Webcam `cv2.VideoCapture()`, gọi bộ AI để phân tích, tính toán FPS mượt mà (bằng đếm số frame trên 1 giây).
*   **Giao tiếp Signals/Slots:** Để chuyển khung hình đã vẽ khung xanh/đỏ từ luồng AI lên luồng Giao diện mà không gây crash, ta dùng cơ chế `pyqtSignal`.

### 3. Thiết kế Giao diện UI/UX (Frontend với PyQt5)
*   **Light Theme & QSS:** Giao diện được thiết kế hiện đại, sáng sủa dựa trên bộ màu tuỳ chỉnh định nghĩa ở `config.py`. Dùng ngôn ngữ stylesheet của Qt (gần giống CSS) để bo góc (border-radius), đổ màu nền, làm nút bấm hình viên thuốc (Pill buttons).
*   **State Management:** Giao diện thay đổi linh hoạt (Header đổi màu, ẩn/hiện Form nhập liệu) dựa trên tab đang được chọn nhờ `QStackedWidget`. Form "Thêm/Sửa" được nhúng trực tiếp vào lưới `QHBoxLayout` tạo cảm giác liền mạch thay vì popup khó chịu.
*   **Chuyển đổi hình ảnh (OpenCV ➡️ Qt):** Ảnh OpenCV là hệ màu `BGR`, trong khi Giao diện PyQt5 hiển thị hệ `RGB`. Chúng ta đã dùng hàm `cv2.cvtColor` để đảo màu và nhúng ảnh thành mảng `QImage / QPixmap` lên giao diện.

### 4. Quản lý Cơ sở dữ liệu (Database & Caching)
*   **SQLite (Ổ cứng):** Lưu trữ thông tin nhân sự (Họ tên, CCCD...). Đặc biệt, mảng 512 số thực (vector) của khuôn mặt được mã hóa thành dữ liệu nhị phân thô (`BLOB`) bằng lệnh `vec.tobytes()` để lưu thẳng vào Database siêu gọn.
*   **VectorStore (RAM Cache):** Đọc cơ sở dữ liệu từ ổ cứng (chậm) lên bộ nhớ RAM (nhanh) thành một ma trận khổng lồ (`np.matrix`). Quá trình này được kích hoạt đồng bộ (Dynamic Reload) mỗi khi bật camera hoặc khi anh bấm Lưu/Xóa người mới bên tab CRUD.

### 5. Kinh nghiệm "Xương máu" xử lý lỗi (Troubleshooting)
Dự án này vượt qua được rất nhiều "căn bệnh" đặc trưng của việc lập trình AI trên Windows:
*   **Lỗi DLL Hell:** Cả `PyQt5`, `PyTorch` (YOLO) và `ONNXRuntime` (InsightFace) đều mang theo bộ thư viện C++ riêng. Nếu nạp không đúng thứ tự, chúng sẽ "đánh nhau" tranh giành tài nguyên. Kỹ thuật giải quyết: Đưa `import onnxruntime` lên dòng số 1 của file `main.py`.
*   **Xung đột CUDA Context:** Khi cả YOLO và InsightFace đều đòi chạy GPU, bộ nhớ card màn hình đôi lúc bị sụp. Kỹ thuật giải quyết: Cô lập, đảo thứ tự Load Model (Load InsightFace trước, YOLO sau), ép InsightFace chạy CPU.

---
Tóm lại, thông qua dự án này anh đang sở hữu một hệ thống tích hợp đầy đủ mọi tinh hoa từ **Computer Vision hiện đại**, **Tối ưu toán học Ma trận**, **Quản trị Luồng (Threading)** cho đến **Thiết kế Giao diện Desktop** hoàn chỉnh!