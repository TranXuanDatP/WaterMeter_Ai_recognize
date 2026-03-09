# Cell Code để thêm vào đầu Notebook Colab
# Copy đoạn code này và thêm vào sau cell "Mount Google Drive"

# ============================================================
# TẠO CẤU TRÚC THỨ MỤC TRÊN GOOGLE DRIVE
# ============================================================

import os

# Cấu trúc thư mục cần tạo
GDRIVE_BASE_DIR = '/content/drive/MyDrive/WaterMeter_Project/M4_Training'
DIRS_TO_CREATE = [
    GDRIVE_BASE_DIR,
    os.path.join(GDRIVE_BASE_DIR, 'data'),
    os.path.join(GDRIVE_BASE_DIR, 'models'),
    os.path.join(GDRIVE_BASE_DIR, 'checkpoints'),
]

print("="*70)
print("TẠO CẤU TRÚC THỨ MỤC TRÊN GOOGLE DRIVE")
print("="*70)

for dir_path in DIRS_TO_CREATE:
    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Đã tạo: {dir_path}")
    except Exception as e:
        print(f"❌ Lỗi khi tạo {dir_path}: {e}")

print("\n" + "="*70)
print("CẤU TRÚC THỨ MỤC SẼ ĐƯỢC TẠO:")
print("="*70)
print("""
Google Drive/
└── MyDrive/
    └── WaterMeter_Project/
        └── M4_Training/
            ├── data/          ← Upload m4_ocr_dataset_black_digits.zip vào đây
            ├── models/        ← Models sẽ được lưu vào đây
            └── checkpoints/   ← Training checkpoints
""")
print("="*70)
print("\n💡 Next step: Upload file m4_ocr_dataset_black_digits.zip")
print(f"   vào thư mục: {os.path.join(GDRIVE_BASE_DIR, 'data')}")
print("="*70)
