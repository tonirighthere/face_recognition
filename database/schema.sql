-- schema.sql — DDL tạo bảng SQLite cho hệ thống nhận diện khuôn mặt

CREATE TABLE IF NOT EXISTS nhan_su (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    ho_ten      TEXT     NOT NULL,
    ngay_sinh   TEXT,                          -- định dạng YYYY-MM-DD
    cccd        TEXT     UNIQUE,               -- Căn cước công dân (duy nhất)
    gioi_tinh   TEXT     CHECK(gioi_tinh IN ('Nam', 'Nữ', 'Khác')),
    dien_thoai  TEXT,
    anh_path    TEXT,                          -- đường dẫn tương đối ảnh đại diện
    embedding   BLOB,                          -- 512-dim float32 vector (numpy bytes)
    created_at  TEXT     DEFAULT (datetime('now', 'localtime')),
    updated_at  TEXT     DEFAULT (datetime('now', 'localtime'))
);

CREATE INDEX IF NOT EXISTS idx_nhan_su_cccd ON nhan_su(cccd);
CREATE INDEX IF NOT EXISTS idx_nhan_su_ho_ten ON nhan_su(ho_ten);
