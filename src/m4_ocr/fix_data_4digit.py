"""
Fix data_4digit.csv - Add Zero-Padding to Value Column
Script này sẽ đọc file data_4digit.csv gốc, thêm zero-padding vào cột value,
 và tạo ra file CSV mới với labels đã được format.
"""
import os
import sys
import codecs
import pandas as pd
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def ensure_4digit_format(value):
    """
    Convert value sang 4-digit string với zero-padding.
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
    if pd.isna(value):
        return "0000"

    value_str = str(value).strip()

    # BƯỚC 2: Remove bất kỳ non-digit characters
    value_str = ''.join(c for c in value_str if c.isdigit())

    # BƯỚC 3: Handle empty string
    if len(value_str) == 0:
        return "0000"

    # BƯỚC 4: Pad với zeros phía trước để đủ 4 chữ số
    # Nếu nhiều hơn 4 chữ số, lấy 4 chữ số cuối
    if len(value_str) > 4:
        value_str = value_str[-4:]

    # Pad với zeros để đủ 4 chữ số
    return value_str.zfill(4)


def fix_data_4digit_csv(input_csv_path, output_csv_path=None):
    """
    Fix file data_4digit.csv với zero-padding

    Args:
        input_csv_path: Đường dẫn đến file data_4digit.csv gốc
        output_csv_path: Đường dẫn file CSV mới (nếu None, tự động tạo)
    """
    print("=" * 70)
    print("FIX data_4digit.csv - ZERO-PADDING")
    print("=" * 70)

    # Validate input file
    if not os.path.exists(input_csv_path):
        print(f"\n❌ ERROR: File không tồn tại: {input_csv_path}")
        return False

    print(f"\n📂 Input file: {input_csv_path}")

    # Tự động tạo output path nếu không cung cấp
    if output_csv_path is None:
        base_dir = os.path.dirname(input_csv_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_csv_path = os.path.join(base_dir, f"data_4digit_fixed_{timestamp}.csv")
        print(f"📝 Output file will be auto-generated")

    print(f"📂 Output file: {output_csv_path}")

    # Read CSV
    try:
        print(f"\n[LOAD] Reading CSV file...")
        df = pd.read_csv(input_csv_path)
        print(f"[LOAD] ✅ Loaded {len(df)} rows")
        print(f"[LOAD] Columns: {list(df.columns)}")
    except Exception as e:
        print(f"\n❌ ERROR: Không thể đọc CSV: {e}")
        return False

    # Check required columns
    if 'value' not in df.columns:
        print(f"\n❌ ERROR: CSV không có cột 'value'")
        print(f"   Available columns: {list(df.columns)}")
        return False

    # Show sample before fixing
    print(f"\n📋 Sample BEFORE fixing:")
    print("-" * 70)
    print(f"{'Photo Name':<50s} | {'Value':<10s} | {'Type'}")
    print("-" * 70)

    for i in range(min(5, len(df))):
        photo_name = df.iloc[i].get('photo_name', 'N/A')
        value = df.iloc[i]['value']
        value_type = type(value).__name__
        print(f"{str(photo_name)[:50]:50s} | {str(value):10s} | {value_type}")

    # Show statistics before fixing
    print(f"\n📊 Statistics BEFORE fixing:")
    print(f"  Total rows: {len(df)}")
    print(f"  Value column type: {df['value'].dtype}")

    # Count value lengths
    value_lengths_before = df['value'].astype(str).str.len()
    print(f"\n  Value length distribution BEFORE:")
    for length in sorted(value_lengths_before.unique()):
        count = (value_lengths_before == length).sum()
        percentage = count / len(df) * 100
        print(f"    {length} digits: {count:5d} ({percentage:5.1f}%)")

    # Apply fixing
    print(f"\n🔧 Applying zero-padding fix...")
    df['value'] = df['value'].apply(ensure_4digit_format)

    # Show sample after fixing
    print(f"\n✅ Sample AFTER fixing:")
    print("-" * 70)
    print(f"{'Photo Name':<50s} | {'Value':<10s} | {'Type'}")
    print("-" * 70)

    for i in range(min(5, len(df))):
        photo_name = df.iloc[i].get('photo_name', 'N/A')
        value = df.iloc[i]['value']
        value_type = type(value).__name__
        print(f"{str(photo_name)[:50]:50s} | {str(value):10s} | {value_type}")

    # Validate results
    print(f"\n📊 VALIDATION RESULTS:")
    print("-" * 70)

    # Check value lengths
    value_lengths = df['value'].apply(len)
    print(f"\nValue length distribution AFTER:")

    all_4digit = True
    for length in sorted(value_lengths.unique()):
        count = (value_lengths == length).sum()
        percentage = count / len(df) * 100
        status = "✅" if length == 4 else "❌"
        print(f"  {status} {length} digits: {count:5d} ({percentage:5.1f}%)")
        if length != 4:
            all_4digit = False

    # Check for non-digit characters
    non_digit_values = df[~df['value'].str.isdigit()]
    if len(non_digit_values) > 0:
        print(f"\n⚠️  WARNING: Found {len(non_digit_values)} values with non-digit characters!")
        print("   First 5 examples:")
        for i, row in non_digit_values.head(5).iterrows():
            print(f"     {row['photo_name'][:50]:50s} | {row['value']}")
    else:
        print(f"\n✅ All values contain only digits")

    # Show value distribution
    print(f"\n📊 Value distribution AFTER fixing:")
    value_ranges = {
        '0000-0999': ((df['value'] >= '0000') & (df['value'] < '1000')).sum(),
        '1000-1999': ((df['value'] >= '1000') & (df['value'] < '2000')).sum(),
        '2000-2999': ((df['value'] >= '2000') & (df['value'] < '3000')).sum(),
        '3000-3999': ((df['value'] >= '3000') & (df['value'] < '4000')).sum(),
        '4000-4999': ((df['value'] >= '4000') & (df['value'] < '5000')).sum(),
        '5000-5999': ((df['value'] >= '5000') & (df['value'] < '6000')).sum(),
        '6000-6999': ((df['value'] >= '6000') & (df['value'] < '7000')).sum(),
        '7000-7999': ((df['value'] >= '7000') & (df['value'] < '8000')).sum(),
        '8000-8999': ((df['value'] >= '8000') & (df['value'] < '9000')).sum(),
        '9000-9999': ((df['value'] >= '9000') & (df['value'] <= '9999')).sum(),
    }

    for range_name, count in value_ranges.items():
        percentage = count / len(df) * 100
        bar = '█' * int(percentage / 5)
        print(f"  {range_name:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

    # Save to new CSV
    print(f"\n💾 Saving to new CSV file...")
    try:
        df.to_csv(output_csv_path, index=False)
        file_size_mb = os.path.getsize(output_csv_path) / (1024 * 1024)
        print(f"✅ Saved to: {output_csv_path}")
        print(f"   File size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ ERROR: Không thể lưu file: {e}")
        return False

    # Final summary
    print(f"\n" + "=" * 70)
    if all_4digit and len(non_digit_values) == 0:
        print("✅ SUCCESS: All values are now correctly formatted as 4-digit strings!")
    else:
        print("⚠️  WARNING: Some issues detected. Please review the validation results above.")

    print(f"\n📁 Output file: {output_csv_path}")
    print(f"   Total rows: {len(df):,}")
    print(f"   All values: 4-digit strings (0000-9999)")
    print("=" * 70)

    return True


def main():
    """Main function"""
    print("\n" + "🎯" * 35)
    print("FIX data_4digit.csv - ZERO-PADDING TOOL")
    print("🎯" * 35)

    # Cấu hình đường dẫn
    INPUT_CSV = r"F:\Workspace\Project\data\data_4digit.csv"
    OUTPUT_CSV = r"F:\Workspace\Project\data\data_4digit_fixed.csv"

    print(f"\n📁 Input file: {INPUT_CSV}")
    print(f"📁 Output file: {OUTPUT_CSV}")

    if not os.path.exists(INPUT_CSV):
        print(f"\n❌ ERROR: File không tồn tại: {INPUT_CSV}")
        print(f"\n💡 Vui lòng cập nhật đường dẫn trong script (biến INPUT_CSV)")
        return

    # Fix CSV
    success = fix_data_4digit_csv(INPUT_CSV, OUTPUT_CSV)

    if success:
        print(f"\n" + "=" * 70)
        print("🎉 SCRIPT COMPLETED!")
        print("=" * 70)
        print(f"\n💡 Next steps:")
        print(f"   1. Kiểm tra file output: {OUTPUT_CSV}")
        print(f"   2. Validate kết quả ở trên")
        print(f"   3. Sử dụng file này để prepare M4 dataset:")
        print(f"      - Update LABELS_FILE trong prepare_dataset.py")
        print(f"      - Chạy: python prepare_dataset.py")
        print("=" * 70)
    else:
        print(f"\n❌ FIX FAILED!")


if __name__ == "__main__":
    main()
