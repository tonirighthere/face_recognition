import os
import sys
import cv2
import logging
import warnings
from pathlib import Path
import numpy as np
import gradio as gr

# Bỏ qua các cảnh báo từ thư viện bên thứ 3
warnings.filterwarnings("ignore", category=FutureWarning)

# Import các config và module từ project
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import APP_NAME, APP_VERSION, STORAGE_DIR, SIMILARITY_THRESHOLD
from core.face_engine import FaceEngine
from models.vector_store import VectorStore
from models.db_manager import DatabaseManager

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("GradioApp")

# Khởi tạo FaceEngine và VectorStore
logger.info("Đang khởi tạo các model AI và Database...")
db_manager = DatabaseManager()
face_engine = FaceEngine.instance()
face_engine.load()
vector_store = VectorStore.instance()
vector_store.load_from_db(db_manager)
logger.info("Khởi tạo hệ thống thành công!")

# ──────── ĐỊNH NGHĨA CÁC HÀM XỬ LÝ ────────

def get_personnel_df():
    """Lấy danh sách nhân sự hiện tại dưới dạng bảng dữ liệu."""
    data = db_manager.get_all()
    if not data:
        return []
    return [
        [p['id'], p['ho_ten'], p['cccd'], p['gioi_tinh'], p['ngay_sinh'], p['dien_thoai']]
        for p in data
    ]

def search_personnel(keyword):
    """Tìm kiếm nhân sự theo Tên hoặc CCCD."""
    if not keyword:
        return get_personnel_df()
    data = db_manager.search(keyword)
    return [
        [p['id'], p['ho_ten'], p['cccd'], p['gioi_tinh'], p['ngay_sinh'], p['dien_thoai']]
        for p in data
    ]

def register_person(ho_ten, ngay_sinh, cccd, gioi_tinh, dien_thoai, image):
    """Đăng ký nhân sự mới, trích xuất embedding và lưu vào DB."""
    if not ho_ten or not cccd:
        return "⚠️ Thất bại: Họ tên và CCCD là các trường bắt buộc!", get_personnel_df()
    if image is None:
        return "⚠️ Thất bại: Vui lòng tải lên hoặc chụp ảnh chân dung!", get_personnel_df()

    # Chuyển đổi ảnh của Gradio (RGB) sang OpenCV (BGR)
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Đếm số khuôn mặt trong ảnh
    face_count = face_engine.count_faces(img_bgr)
    if face_count == 0:
        return "❌ Thất bại: Không tìm thấy khuôn mặt nào trong ảnh. Hãy chọn ảnh rõ mặt hơn!", get_personnel_df()
    elif face_count > 1:
        return f"❌ Thất bại: Phát hiện {face_count} khuôn mặt. Chỉ chấp nhận ảnh có duy nhất 1 khuôn mặt!", get_personnel_df()

    # Trích xuất vector đặc trưng (embedding)
    emb = face_engine.get_embedding_from_image(img_bgr)
    if emb is None:
        return "❌ Thất bại: Không thể trích xuất đặc trưng khuôn mặt (ảnh chất lượng thấp hoặc quá mờ)!", get_personnel_df()

    # Lưu ảnh vào thư mục storage/photos
    photo_name = f"{cccd}.jpg"
    photo_path = STORAGE_DIR / photo_name
    cv2.imwrite(str(photo_path), img_bgr)

    try:
        # Lưu thông tin vào database SQLite
        person_id = db_manager.add_person(
            ho_ten=ho_ten,
            ngay_sinh=ngay_sinh,
            cccd=cccd,
            gioi_tinh=gioi_tinh,
            dien_thoai=dien_thoai,
            anh_path=str(photo_path),
            embedding=emb
        )
        
        # Thêm nóng vào VectorStore RAM cache
        vector_store.add(person_id, ho_ten, emb)
        
        return f"✅ Thành công: Đã đăng ký nhân sự '{ho_ten}' (ID: {person_id})", get_personnel_df()
    except Exception as e:
        return f"❌ Thất bại: Lỗi khi lưu vào cơ sở dữ liệu: {e}", get_personnel_df()

def delete_person_by_id(person_id):
    """Xóa nhân sự theo ID."""
    if not person_id:
        return "⚠️ Vui lòng cung cấp ID nhân sự cần xóa!", get_personnel_df()
    
    try:
        p_id = int(person_id)
    except ValueError:
        return "⚠️ ID nhân sự phải là một số nguyên hợp lệ!", get_personnel_df()

    person = db_manager.get_by_id(p_id)
    if not person:
        return f"❌ Thất bại: Không tìm thấy nhân sự có ID = {p_id}!", get_personnel_df()

    # Xóa file ảnh lưu trữ
    if person['anh_path'] and Path(person['anh_path']).exists():
        try:
            Path(person['anh_path']).unlink()
        except Exception as e:
            logger.warning(f"Không thể xóa file ảnh vật lý: {e}")

    # Xóa trong database
    success = db_manager.delete_person(p_id)
    if success:
        # Nạp lại vector store để xóa cache RAM
        vector_store.load_from_db(db_manager)
        return f"✅ Thành công: Đã xóa nhân sự '{person['ho_ten']}' (ID: {p_id}) khỏi hệ thống.", get_personnel_df()
    else:
        return "❌ Thất bại: Lỗi cơ sở dữ liệu khi thực hiện xóa.", get_personnel_df()

def recognize_faces(image, apply_filter, threshold):
    """Nhận diện khuôn mặt từ frame đầu vào."""
    if image is None:
        return None, "Vui lòng chụp webcam hoặc tải lên hình ảnh."

    # Chuyển ảnh Gradio (RGB) sang OpenCV (BGR)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect khuôn mặt
    bboxes = face_engine.detect(frame, apply_filter=apply_filter)
    
    recognized_names = []
    
    for bbox in bboxes:
        x1, y1, x2, y2, score = bbox
        # Trích xuất vector embedding
        emb = face_engine.get_embedding(frame, bbox)
        
        name = "Unknown"
        sim = 0.0
        
        if emb is not None:
            # So khớp vector với bộ nhớ RAM cache
            res = vector_store.search_best(emb, threshold=threshold)
            if res is not None:
                pid, ho_ten, sim = res
                name = ho_ten
                recognized_names.append(f"👤 {ho_ten} (Độ tương đồng: {sim:.2f})")
            else:
                recognized_names.append("👤 Người lạ (Unknown)")
        else:
            recognized_names.append("⚠️ Ảnh quá mờ/Lỗi trích xuất")

        # Vẽ bounding box và hiển thị tên lên ảnh
        color = (0, 230, 100) if name != "Unknown" else (80, 80, 255) # BGR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label_text = f"{name}" if name == "Unknown" else f"{name} ({sim:.2f})"
        # Vẽ background cho text để dễ đọc
        (w_txt, h_txt), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + w_txt, y1), color, -1)
        cv2.putText(frame, label_text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Chuyển ảnh OpenCV (BGR) về lại RGB để hiển thị trên Gradio
    output_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if not recognized_names:
        result_text = "Không phát hiện khuôn mặt nào hợp lệ."
    else:
        result_text = "\n".join(recognized_names)

    return output_rgb, result_text

# ──────── GIAO DIỆN GRADIO ────────

# Cấu hình theme hiện đại
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title=f"{APP_NAME} - Web Demo") as demo:
    
    gr.Markdown(
        f"""
        # 🚀 {APP_NAME} (Web Demo)
        Hệ thống nhận diện khuôn mặt Realtime sử dụng **MediaPipe** (Face Detection), **InsightFace** (Face Recognition) & **SQLite** làm cơ sở dữ liệu.
        Chạy mượt mà trên CPU, được đóng gói bằng Docker để dễ dàng bàn giao và show demo cho khách hàng.
        """
    )
    
    with gr.Tabs():
        
        # TAB 1: NHẬN DIỆN KHUÔN MẶT
        with gr.TabItem("🎥 Nhận diện"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Đầu vào")
                    input_type = gr.Radio(
                        ["Upload Ảnh", "Webcam"], 
                        label="Chọn nguồn camera/ảnh", 
                        value="Upload Ảnh"
                    )
                    
                    # File upload
                    upload_img = gr.Image(label="Tải ảnh lên", type="numpy", visible=True)
                    # Webcam stream
                    webcam_img = gr.Image(label="Chụp từ Webcam", type="numpy", sources="webcam", visible=False)
                    
                    # Thay đổi hiển thị nguồn đầu vào
                    def update_input_source(choice):
                        if choice == "Webcam":
                            return gr.update(visible=False), gr.update(visible=True)
                        else:
                            return gr.update(visible=True), gr.update(visible=False)
                            
                    input_type.change(
                        update_input_source, 
                        inputs=input_type, 
                        outputs=[upload_img, webcam_img]
                    )
                    
                    with gr.Accordion("⚙️ Cấu hình bộ lọc & Ngưỡng", open=False):
                        apply_filter = gr.Checkbox(
                            label="Bật bộ lọc chất lượng khuôn mặt (Blur/Roll/Yaw/Pitch)", 
                            value=False
                        )
                        threshold = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0, 
                            value=SIMILARITY_THRESHOLD, 
                            step=0.05, 
                            label="Ngưỡng so khớp (Cosine Similarity)"
                        )
                    
                    btn_recognize = gr.Button("🔍 Bắt đầu nhận diện", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Kết quả")
                    output_img = gr.Image(label="Ảnh kết quả", type="numpy")
                    output_text = gr.Textbox(label="Danh tính phát hiện", placeholder="Kết quả nhận diện sẽ xuất hiện ở đây...", interactive=False)
            
            # Kết nối các nút xử lý nhận diện
            btn_recognize.click(
                fn=recognize_faces,
                inputs=[upload_img, apply_filter, threshold],
                outputs=[output_img, output_text]
            )
            # Sự kiện click nhận diện webcam
            webcam_img.change(
                fn=recognize_faces,
                inputs=[webcam_img, apply_filter, threshold],
                outputs=[output_img, output_text]
            )
            
        # TAB 2: QUẢN LÝ NHÂN SỰ (CRUD)
        with gr.TabItem("👤 Quản lý Nhân sự"):
            with gr.Row():
                # Cột đăng ký nhân sự mới
                with gr.Column(scale=1):
                    gr.Markdown("### 🆕 Đăng ký Nhân sự mới")
                    reg_name = gr.Textbox(label="Họ và Tên (*)", placeholder="Ví dụ: Nguyễn Văn A")
                    reg_cccd = gr.Textbox(label="Số CCCD / Mã Nhân viên (*)", placeholder="Ví dụ: 001202000123")
                    reg_gender = gr.Dropdown(["Nam", "Nữ", "Khác"], label="Giới tính", value="Nam")
                    reg_dob = gr.Textbox(label="Ngày sinh", placeholder="Ví dụ: 1995-10-25")
                    reg_phone = gr.Textbox(label="Số điện thoại", placeholder="Ví dụ: 0987654321")
                    reg_image = gr.Image(label="Ảnh chân dung đăng ký (*)", type="numpy", sources=["upload", "webcam"])
                    
                    reg_btn = gr.Button("💾 Lưu nhân sự", variant="primary")
                    reg_status = gr.Textbox(label="Trạng thái đăng ký", interactive=False)
                    
                # Cột danh sách và xóa nhân sự
                with gr.Column(scale=2):
                    gr.Markdown("### 📋 Danh sách Nhân sự hệ thống")
                    with gr.Row():
                        search_bar = gr.Textbox(
                            label="Tìm kiếm nhanh", 
                            placeholder="Nhập tên hoặc số CCCD...", 
                            scale=4
                        )
                        btn_refresh = gr.Button("🔄 Tải lại", scale=1)
                        
                    personnel_table = gr.Dataframe(
                        headers=["ID", "Họ Tên", "CCCD", "Giới Tính", "Ngày Sinh", "Điện Thoại"],
                        datatype=["number", "str", "str", "str", "str", "str"],
                        value=get_personnel_df(),
                        label="Dữ liệu nhân sự",
                        interactive=False
                    )
                    
                    gr.Markdown("### 🗑️ Xóa nhân sự")
                    with gr.Row():
                        delete_id = gr.Textbox(
                            label="ID nhân sự muốn xóa", 
                            placeholder="Ví dụ: 12",
                            scale=3
                        )
                        delete_btn = gr.Button("❌ Xóa nhân viên", variant="stop", scale=2)
                    delete_status = gr.Textbox(label="Trạng thái xóa", interactive=False)

            # Các sự kiện tab CRUD
            reg_btn.click(
                fn=register_person,
                inputs=[reg_name, reg_dob, reg_cccd, reg_gender, reg_phone, reg_image],
                outputs=[reg_status, personnel_table]
            ).then(
                fn=lambda: (gr.update(value=""), gr.update(value=""), gr.update(value="Nam"), gr.update(value=""), gr.update(value=""), gr.update(value=None)),
                outputs=[reg_name, reg_cccd, reg_gender, reg_dob, reg_phone, reg_image]
            )
            
            search_bar.change(
                fn=search_personnel,
                inputs=search_bar,
                outputs=personnel_table
            )
            
            btn_refresh.click(
                fn=get_personnel_df,
                outputs=personnel_table
            )
            
            delete_btn.click(
                fn=delete_person_by_id,
                inputs=delete_id,
                outputs=[delete_status, personnel_table]
            ).then(
                fn=lambda: gr.update(value=""),
                outputs=delete_id
            )

# Chạy ứng dụng Gradio
if __name__ == "__main__":
    # Gradio sẽ lắng nghe ở port 7860
    # Thiết lập server_name="0.0.0.0" để Docker container có thể map port ra máy chủ
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
