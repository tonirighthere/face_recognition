# 🚀 Face Recognition Desktop App (Pipeline Guide)

## 🎯 Mục tiêu
Xây dựng ứng dụng Desktop:
- Nhận diện khuôn mặt realtime (Liveview)
- Quản lý người (CRUD)

---

## 🏗️ 1. Kiến trúc hệ thống

UI (PyQt5)  
↓  
Controller  
↓  
AI Worker (QThread)  
↓  
Face Engine  
↓  
Vector Search (RAM)  
↓  
SQLite + Storage  

---

## 🔴 2. Pipeline Liveview

Camera → Frame Skip → Detect → Track → Align → Embed → Search → Render

### Chi tiết:
1. Webcam stream (OpenCV)
2. Frame skipping (mỗi 3 frame chạy AI)
3. Detect face (YOLOv8-face)
4. Tracking (ByteTrack)
5. Face alignment (InsightFace)
6. Embedding (512D vector)
7. Vector search (NumPy/FAISS)
8. Hiển thị UI

---

## 🔵 3. Pipeline CRUD

Input → Validate → Detect (1 face) → Align → Embed → Save DB → Reload cache

---

## 🧱 4. Database

Table: nhan_su

- id (PK)
- ho_ten
- ngay_sinh
- cccd (UNIQUE)
- gioi_tinh
- dien_thoai
- anh_path
- embedding (BLOB)

---

## ⚡ 5. Performance

- Frame skipping
- Tracking
- RAM cache embedding
- Batch cosine similarity

---

## 🧩 6. Project Structure

project/
├── app/
├── core/
├── database/
├── storage/
└── config.py

---

## 🧠 7. Tech Stack

- PyQT5
- OpenCV
- YOLOv8-face
- InsightFace (ArcFace)
- NumPy / FAISS
- SQLite

---

## ✅ 8. Roadmap

### Phase 1
- Detect face
- Extract embedding
- Cosine similarity

### Phase 2
- Multi-user recognition

### Phase 3
- CRUD + SQLite

### Phase 4
- UI PyQt5

### Phase 5
- Tracking + optimization

---
