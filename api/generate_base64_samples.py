#!/usr/bin/env python3
"""
Convert images to Base64 format for API testing
"""
import base64
import json
from pathlib import Path
from typing import List, Dict


def image_to_base64(image_path: Path) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def create_data_url(image_path: Path, base64_string: str) -> str:
    """Create data URL format"""
    # Get mime type
    ext = image_path.suffix.lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }.get(ext, 'image/jpeg')

    return f"data:{mime_type};base64,{base64_string}"


def get_true_value(image_name: str) -> str:
    """Get true value from CSV if available"""
    csv_path = Path(r"F:\Workspace\Project\data\images_4digit2.csv")

    if not csv_path.exists():
        return None

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df['photo_name'] == image_name]
        if len(row) > 0:
            return str(row.iloc[0]['value'])
    except:
        pass

    return None


def generate_base64_samples(
    data_dir: str,
    output_file: str,
    num_samples: int = 10,
    format: str = "both"  # "raw", "data_url", or "both"
):
    """
    Generate base64 samples from images

    Args:
        data_dir: Directory containing images
        output_file: Output JSON file path
        num_samples: Number of images to convert
        format: Output format - "raw", "data_url", or "both"
    """
    print("="*70)
    print("IMAGE TO BASE64 CONVERTER")
    print("="*70)

    data_path = Path(data_dir)

    # Check if directory exists
    if not data_path.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return

    # Find images
    print(f"\n[*] Scanning images from: {data_dir}")
    image_files = list(data_path.glob("*.jpg")) + list(data_path.glob("*.png"))

    if not image_files:
        print(f"✗ No images found")
        return

    print(f"  ✓ Found {len(image_files)} images")
    print(f"  ✓ Converting first {min(num_samples, len(image_files))} images")

    # Convert images
    samples = []

    for i, img_path in enumerate(image_files[:num_samples], 1):
        print(f"\n[{i}/{min(num_samples, len(image_files))}] {img_path.name}")

        try:
            # Convert to base64
            base64_raw = image_to_base64(img_path)
            base64_data_url = create_data_url(img_path, base64_raw)

            # Get true value if available
            true_value = get_true_value(img_path.name)

            sample = {
                'filename': img_path.name,
                'filepath': str(img_path),
                'true_value': true_value,
                'size_bytes': len(base64_raw) * 3 // 4,
                'size_kb': round(len(base64_raw) * 3 / 4 / 1024, 2)
            }

            # Add base64 based on format preference
            if format in ["raw", "both"]:
                sample['base64_raw'] = base64_raw
            if format in ["data_url", "both"]:
                sample['base64_data_url'] = base64_data_url

            samples.append(sample)

            print(f"  ✓ Converted: {sample['size_kb']} KB")
            if true_value:
                print(f"  ✓ True value: {true_value}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Saving to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(samples)} samples")

    # Generate usage examples
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)

    print("\n1. Test with single sample (Raw Base64):")
    print(f"   >>> import json")
    print(f"   >>> data = json.load(open('{output_file}'))")
    print(f"   >>> base64_str = data[0]['base64_raw']")
    print(f"   >>> # Use this base64_str in API request")

    print("\n2. Test with Data URL format:")
    print(f"   >>> import json")
    print(f"   >>> data = json.load(open('{output_file}'))")
    print(f"   >>> data_url = data[0]['base64_data_url']")

    print("\n3. Quick API test example:")
    print(f"   >>> import requests, json")
    print(f"   >>> data = json.load(open('{output_file}'))")
    print(f"   >>> response = requests.post(")
    print(f"   ...     'http://localhost:8000/predict',")
    print(f"   ...     json={{'image': data[0]['base64_raw']}}")
    print(f"   ... )")
    print(f"   >>> print(response.json())")

    print("\n4. Use with test_base64_direct.py:")
    print(f"   # Create a text file with base64 content")
    print(f"   >>> with open('test_sample.txt', 'w') as f:")
    print(f"   ...     f.write(data[0]['base64_raw'])")
    print(f"   >>> # Then test:")
    print(f"   $ python test_base64_direct.py test_sample.txt")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Total samples: {len(samples)}")
    print(f"  Output file: {output_file}")
    print(f"  Format: {format}")

    if len(samples) > 0:
        total_size = sum(s['size_kb'] for s in samples)
        avg_size = total_size / len(samples)
        print(f"  Total size: {total_size:.2f} KB")
        print(f"  Avg size: {avg_size:.2f} KB")

    print("="*70)

    return samples


def main():
    # Configuration
    DATA_DIR = r"F:\Workspace\Project\data\data_4digit2"
    OUTPUT_FILE = r"F:\Workspace\Project\api\test_samples_base64.json"
    NUM_SAMPLES = 10  # Convert first 10 images

    # Generate base64 samples
    generate_base64_samples(
        data_dir=DATA_DIR,
        output_file=OUTPUT_FILE,
        num_samples=NUM_SAMPLES,
        format="both"  # Generate both raw and data_url formats
    )


if __name__ == "__main__":
    main()
