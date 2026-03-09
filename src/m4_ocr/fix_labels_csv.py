"""
Fix Labels CSV - Add Zero-Padding to Ensure 4-Digit Format
Script này sẽ fix file labels.csv hiện tại, đảm bảo tất cả labels có đúng 4 chữ số.
"""
import os
import sys
import codecs
import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def ensure_4digit_format(label):
    """
    Convert label sang 4-digit string với zero-padding.
    QUAN TRỌNG: Convert sang string TRƯỚC khi xử lý

    Examples:
        5 -> '0005'
        87 -> '0087'
        187 -> '0187'
        1234 -> '1234'
        '5' -> '0005'
        '187' -> '0187'
    """
    # BƯỚC 1: Convert sang string (quan trọng nhất!)
    label_str = str(label).strip()

    # BƯỚC 2: Remove bất kỳ non-digit characters
    label_str = ''.join(c for c in label_str if c.isdigit())

    # BƯỜC 3: Handle empty string
    if len(label_str) == 0:
        return "0000"

    # BƯỚC 4: Pad với zeros phía trước để đủ 4 chữ số
    # Nếu nhiều hơn 4 chữ số, lấy 4 chữ số cuối
    if len(label_str) > 4:
        label_str = label_str[-4:]

    # Pad với zeros để đủ 4 chữ số
    return label_str.zfill(4)


def fix_labels_file(labels_csv_path, backup=True):
    """
    Fix labels trong CSV file

    Args:
        labels_csv_path: Đường dẫn đến file labels.csv
        backup: Có tạo backup không (default: True)
    """
    print("=" * 70)
    print("FIX LABELS CSV - ZERO-PADDING")
    print("=" * 70)

    # Validate input file
    if not os.path.exists(labels_csv_path):
        print(f"\n❌ ERROR: File không tồn tại: {labels_csv_path}")
        return False

    print(f"\n📂 Processing: {labels_csv_path}")

    # Backup original file
    if backup:
        backup_path = f"{labels_csv_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(labels_csv_path, backup_path)
        print(f"💾 Backup created: {backup_path}")

    # Read CSV
    try:
        df = pd.read_csv(labels_csv_path)
        print(f"\n📊 Loaded {len(df)} rows from CSV")
        print(f"   Columns: {list(df.columns)}")
    except Exception as e:
        print(f"\n❌ ERROR: Không thể đọc CSV: {e}")
        return False

    # Check required columns
    if 'text' not in df.columns:
        print(f"\n❌ ERROR: CSV không có cột 'text'")
        print(f"   Available columns: {list(df.columns)}")
        return False

    # Show sample before fixing
    print(f"\n📋 Sample labels BEFORE fixing:")
    print("-" * 70)
    for i in range(min(5, len(df))):
        filename = df.iloc[i].get('filename', 'N/A')
        text = df.iloc[i]['text']
        print(f"  {str(filename)[:50]:50s} | {str(text):10s} (type: {type(text).__name__})")

    # Apply fixing
    print(f"\n🔧 Applying zero-padding fix...")
    df['text'] = df['text'].apply(ensure_4digit_format)

    # Show sample after fixing
    print(f"\n✅ Sample labels AFTER fixing:")
    print("-" * 70)
    for i in range(min(5, len(df))):
        filename = df.iloc[i].get('filename', 'N/A')
        text = df.iloc[i]['text']
        print(f"  {str(filename)[:50]:50s} | {str(text):10s} (type: {type(text).__name__})")

    # Validate results
    print(f"\n📊 VALIDATION RESULTS:")
    print("-" * 70)

    # Check label lengths
    label_lengths = df['text'].apply(len)
    print(f"\nLabel length distribution:")
    all_4digit = True
    for length in sorted(label_lengths.unique()):
        count = (label_lengths == length).sum()
        percentage = count / len(df) * 100
        status = "✅" if length == 4 else "❌"
        print(f"  {status} {length} digits: {count:5d} ({percentage:5.1f}%)")
        if length != 4:
            all_4digit = False

    # Check for non-digit characters
    non_digit_labels = df[~df['text'].str.isdigit()]
    if len(non_digit_labels) > 0:
        print(f"\n⚠️  WARNING: Found {len(non_digit_labels)} labels with non-digit characters!")
        print("   First 5 examples:")
        for i, row in non_digit_labels.head(5).iterrows():
            print(f"     {row['filename'][:50]:50s} | {row['text']}")
    else:
        print(f"\n✅ All labels contain only digits")

    # Save fixed labels
    print(f"\n💾 Saving fixed labels...")
    df.to_csv(labels_csv_path, index=False)
    print(f"✅ Saved to: {labels_csv_path}")

    # Final summary
    print(f"\n" + "=" * 70)
    if all_4digit and len(non_digit_labels) == 0:
        print("✅ SUCCESS: All labels are now correctly formatted as 4-digit strings!")
    else:
        print("⚠️  WARNING: Some issues detected. Please review the validation results above.")
    print("=" * 70)

    return True


def main():
    """Main function"""
    print("\n" + "🎯" * 35)
    print("FIX LABELS CSV - ZERO-PADDING TOOL")
    print("🎯" * 35)

    # Cấu hình đường dẫn
    # Thay đổi đường dẫn này đến file labels.csv của bạn
    LABELS_CSV = r"F:\Workspace\Project\data\m4_ocr_dataset_black_digits\labels.csv"

    print(f"\n📁 Target file: {LABELS_CSV}")

    if not os.path.exists(LABELS_CSV):
        print(f"\n❌ ERROR: File không tồn tại: {LABELS_CSV}")
        print(f"\n💡 Vui lòng cập nhật đường dẫn trong script (biến LABELS_CSV)")
        return

    # Fix labels
    success = fix_labels_file(LABELS_CSV, backup=True)

    if success:
        print(f"\n" + "=" * 70)
        print("🎉 SCRIPT COMPLETED!")
        print("=" * 70)
        print(f"\n💡 Next steps:")
        print(f"   1. Kiểm tra kết quả validation ở trên")
        print(f"   2. Nếu labels đã đúng, ZIP lại dataset")
        print(f"   3. Upload lên Google Drive")
        print(f"   4. Run training notebook trong Colab")
        print("=" * 70)
    else:
        print(f"\n❌ FIX FAILED!")


if __name__ == "__main__":
    main()
