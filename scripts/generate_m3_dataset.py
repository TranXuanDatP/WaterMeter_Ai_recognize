"""
Generate M3 Dataset: Process all images through M1-M2-M3 pipeline
Save counter ROI images after M3 step
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.m2_orientation_alignment.inference import M2Inference


def process_single_image(img_path: Path, m1_model, m2_inference, m3_model):
    """
    Process single image through M1-M2-M3 pipeline

    Returns:
        dict: {
            'success': bool,
            'counter_roi': np.ndarray or None,
            'metadata': dict
        }
    """
    try:
        # Load original image
        original = cv2.imread(str(img_path))
        if original is None:
            return {
                'success': False,
                'counter_roi': None,
                'metadata': {'error': 'Cannot load image'}
            }

        h_orig, w_orig = original.shape[:2]

        # M1: Watermeter Detection
        m1_results = m1_model(original, verbose=False)
        if len(m1_results) == 0 or len(m1_results[0].boxes) == 0:
            return {
                'success': False,
                'counter_roi': None,
                'metadata': {'error': 'M1: No watermeter detected'}
            }

        boxes = m1_results[0].boxes
        best_idx = boxes.conf.argmax()
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf_m1 = float(boxes.conf[best_idx])

        meter_crop = original[y1:y2, x1:x2]

        # M2: Orientation Alignment
        try:
            m2_result = m2_inference.align_with_info(meter_crop)
            meter_aligned = m2_result['aligned_image']
            detected_angle = m2_result['detected_angle']
            correction_angle = m2_result['correction_angle']
        except Exception as e:
            # If M2 fails, use original crop
            meter_aligned = meter_crop
            detected_angle = 0.0
            correction_angle = 0.0

        # M3: Counter Detection
        m3_results = m3_model(meter_aligned, verbose=False)
        if len(m3_results) == 0 or len(m3_results[0].boxes) == 0:
            return {
                'success': False,
                'counter_roi': None,
                'metadata': {'error': 'M3: No counter detected'}
            }

        boxes = m3_results[0].boxes
        best_idx = boxes.conf.argmax()
        cx1, cy1, cx2, cy2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        conf_m3 = float(boxes.conf[best_idx])

        counter_roi = meter_aligned[cy1:cy2, cx1:cx2]

        metadata = {
            'original_size': (w_orig, h_orig),
            'm1_bbox': (int(x1), int(y1), int(x2), int(y2)),
            'm1_confidence': float(conf_m1),
            'm2_detected_angle': float(detected_angle),
            'm2_correction_angle': float(correction_angle),
            'm3_bbox': (int(cx1), int(cy1), int(cx2), int(cy2)),
            'm3_confidence': float(conf_m3),
            'counter_roi_size': (counter_roi.shape[1], counter_roi.shape[0])
        }

        return {
            'success': True,
            'counter_roi': counter_roi,
            'metadata': metadata
        }

    except Exception as e:
        return {
            'success': False,
            'counter_roi': None,
            'metadata': {'error': f'Processing error: {str(e)}'}
        }


def main():
    # Configuration
    input_dir = Path("data/images_4digit")
    output_dir = Path("data/m3_dataset")
    output_dir.mkdir(exist_ok=True)

    # Create subdirectories
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Load models
    print("=" * 60)
    print("M3 Dataset Generation")
    print("=" * 60)
    print("\nLoading models...")
    m1_model = YOLO("model/detect_watermeter.pt")
    m2_inference = M2Inference("model/orientation.pth", device='cpu')
    m3_model = YOLO("model/detect_array.pt")
    print(">> All models loaded\n")

    # Get all images
    image_files = sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png"))
    total_images = len(image_files)
    print(f"Found {total_images} images to process\n")

    # Processing statistics
    stats = {
        'total': total_images,
        'success': 0,
        'failed_m1': 0,
        'failed_m3': 0,
        'failed_other': 0,
        'errors': []
    }

    # Process each image
    results_data = []

    for img_path in tqdm(image_files, desc="Processing images"):
        result = process_single_image(img_path, m1_model, m2_inference, m3_model)

        if result['success']:
            # Save counter ROI
            output_filename = img_path.stem + "_m3.jpg"
            output_path = images_dir / output_filename
            cv2.imwrite(str(output_path), result['counter_roi'])

            # Store metadata
            results_data.append({
                'source_image': str(img_path.name),
                'output_image': output_filename,
                'status': 'success',
                **result['metadata']
            })
            stats['success'] += 1
        else:
            error = result['metadata'].get('error', 'Unknown error')
            results_data.append({
                'source_image': str(img_path.name),
                'status': 'failed',
                'error': error
            })

            # Track error types
            if 'M1' in error:
                stats['failed_m1'] += 1
            elif 'M3' in error:
                stats['failed_m3'] += 1
            else:
                stats['failed_other'] += 1
                stats['errors'].append({'image': str(img_path.name), 'error': error})

    # Save metadata JSON
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'results': results_data
        }, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nStatistics:")
    print(f"  Total images:     {stats['total']}")
    print(f"  Successful:       {stats['success']} ({100*stats['success']/stats['total']:.1f}%)")
    print(f"  Failed (M1):      {stats['failed_m1']} ({100*stats['failed_m1']/stats['total']:.1f}%)")
    print(f"  Failed (M3):      {stats['failed_m3']} ({100*stats['failed_m3']/stats['total']:.1f}%)")
    print(f"  Failed (Other):   {stats['failed_other']} ({100*stats['failed_other']/stats['total']:.1f}%)")
    print(f"\nDataset saved to: {output_dir.absolute()}")
    print(f"  - Images:         {images_dir}")
    print(f"  - Metadata:       {metadata_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
