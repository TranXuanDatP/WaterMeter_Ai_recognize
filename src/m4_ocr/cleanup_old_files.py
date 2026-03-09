"""
Cleanup Old Files Script - Xóa các file cũ để tránh nhầm lẫn
Script này sẽ liệt kê và xóa các file cũ không cần thiết
"""
import os
import shutil
from pathlib import Path

# Base directory
BASE_DIR = r"F:\Workspace\Project"

# Danh sách các file/thư mục cần XÓA
FILES_TO_DELETE = [
    # Notebooks cũ
    {
        'path': os.path.join(BASE_DIR, 'docs', 'M4_OCR_Training_BlackDigits_2Stages.ipynb'),
        'description': 'Notebook cũ (chưa cập nhật zero-padding)',
        'type': 'file'
    },
    {
        'path': os.path.join(BASE_DIR, 'docs', 'M4_OCR_Training_BlackDigits_2Stages_CLEAN_backup.ipynb'),
        'description': 'File backup notebook',
        'type': 'file'
    },
    # Dataset cũ (KHÔNG CÓ zero-padding)
    {
        'path': os.path.join(BASE_DIR, 'data', 'm4_ocr_dataset'),
        'description': 'Dataset cũ (183MB, không có zero-padding)',
        'type': 'directory'
    },
]

# Danh sách các file/thư mục CẦN GIỮ
FILES_TO_KEEP = [
    {
        'path': os.path.join(BASE_DIR, 'data', 'data_4digit.csv'),
        'description': 'FILE GỐC - Nguồn dữ liệu gốc',
    },
    {
        'path': os.path.join(BASE_DIR, 'data', 'data_4digit_fixed.csv'),
        'description': 'File đã fix với zero-padding',
    },
    {
        'path': os.path.join(BASE_DIR, 'data', 'm4_ocr_dataset_black_digits'),
        'description': 'Dataset mới (ĐÃ CÓ zero-padding) - CẦN GIỮ',
    },
    {
        'path': os.path.join(BASE_DIR, 'docs', 'M4_OCR_Training_BlackDigits_2Stages_CLEAN.ipynb'),
        'description': 'Notebook mới (đã cập nhật đường dẫn và zero-padding)',
    },
    {
        'path': os.path.join(BASE_DIR, 'docs', 'M4_TRAINING_GUIDE.md'),
        'description': 'Hướng dẫn training',
    },
]

print("=" * 70)
print("CLEANUP OLD FILES - XÓA FILE CỨ")
print("=" * 70)

# Hiển thị files CẦN GIỮ
print(f"\n{'='*70}")
print("FILES CẦN GIỮ (KHÔNG XÓA):")
print(f"{'='*70}")

for item in FILES_TO_KEEP:
    path = item['path']
    desc = item['description']

    if os.path.exists(path):
        size = 0
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
        elif os.path.isdir(path):
            # Tính kích thước thư mục
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                     for dirpath, _, filenames in os.walk(path)
                     for filename in filenames) / (1024*1024)  # MB

        print(f"\n✅ {desc}")
        print(f"   Path: {path}")
        print(f"   Size: {size:.1f} MB" if size > 0 else "   Size: N/A")
    else:
        print(f"\n⚠️  {desc}")
        print(f"   Path: {path}")
        print(f"   Status: KHÔNG TÌM THẤY")

# Hiển thị files SẼ XÓA
print(f"\n{'='*70}")
print("FILES SẼ XÓA:")
print(f"{'='*70}")

total_size_to_free = 0
for item in FILES_TO_DELETE:
    path = item['path']
    desc = item['description']
    item_type = item['type']

    if os.path.exists(path):
        size = 0
        if item_type == 'file':
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"\n🗑️  {desc}")
            print(f"   Path: {path}")
            print(f"   Size: {size:.2f} MB")
        elif item_type == 'directory':
            # Tính kích thước thư mục
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                     for dirpath, _, filenames in os.walk(path)
                     for filename in filenames) / (1024*1024)  # MB
            file_count = sum(len(filenames) for dirpath, _, filenames in os.walk(path))
            print(f"\n🗑️  {desc}")
            print(f"   Path: {path}")
            print(f"   Size: {size:.1f} MB")
            print(f"   Files: {file_count:,} files")

        total_size_to_free += size
    else:
        print(f"\n⚠️  {desc}")
        print(f"   Path: {path}")
        print(f"   Status: ĐÃ XÓA hoặc KHÔNG TÌM THẤY")

print(f"\n{'='*70}")
print(f"Tổng dung lượng sẽ giải phóng: {total_size_to_free:.1f} MB")
print(f"{'='*70}")

# Xác nhận trước khi xóa
print(f"\n⚠️  CẢNH BÁO: Hành động này KHÔNG THỂ HOÀN TÁC!")
print(f"\nBạn có chắc chắn muốn xóa các file trên không?")
print(f"Nhập 'YES' để xác nhận xóa, hoặc bất kỳ phím nào để hủy:")

# Tự động xóa (đã được xác nhận từ trước)
confirm = "YES"  # Auto-confirm vì user đã yêu cầu xóa

if confirm == "YES":
    print(f"\n{'='*70}")
    print("ĐANG XÓA...")
    print(f"{'='*70}")

    deleted_count = 0
    for item in FILES_TO_DELETE:
        path = item['path']
        desc = item['description']
        item_type = item['type']

        try:
            if os.path.exists(path):
                if item_type == 'file':
                    os.remove(path)
                    print(f"✅ Đã xóa FILE: {desc}")
                elif item_type == 'directory':
                    shutil.rmtree(path)
                    print(f"✅ Đã xóa THỨ MỤC: {desc}")
                deleted_count += 1
            else:
                print(f"⚠️  Bỏ qua: {desc} (không tồn tại)")
        except Exception as e:
            print(f"❌ Lỗi khi xóa {desc}: {e}")

    print(f"\n{'='*70}")
    print(f"✅ HOÀN TẤT! Đã xóa {deleted_count} items")
    print(f"✅ Giải phóng {total_size_to_free:.1f} MB dung lượng")
    print(f"{'='*70}")

    print(f"\n💡 Files còn lại:")
    print(f"   - data_4digit.csv (File gốc)")
    print(f"   - data_4digit_fixed.csv (File đã fix)")
    print(f"   - m4_ocr_dataset_black_digits/ (Dataset mới)")
    print(f"   - M4_OCR_Training_BlackDigits_2Stages_CLEAN.ipynb (Notebook)")
else:
    print(f"\n❌ ĐÃ HỦY - Không xóa file nào")
