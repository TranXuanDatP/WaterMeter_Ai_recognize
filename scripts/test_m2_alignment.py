"""
Test M2 Alignment - Visualize Results

This script tests M2 orientation alignment on sample images and visualizes:
- Original cropped image (from M1)
- Detected angle
- Aligned (upright) image
- Side-by-side comparison
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.m2_orientation_alignment.inference import M2Inference


def test_m2_on_samples(
    model_path: str = "model/orientation.pth",
    num_samples: int = 5,
    output_dir: str = "test_m2_results"
):
    """
    Test M2 alignment on sample images.

    Args:
        model_path: Path to M2 model
        num_samples: Number of samples to test
        output_dir: Directory to save results
    """
    # Initialize M2
    print("Loading M2 model...")
    aligner = M2Inference(model_path, device='cpu')
    print(">> M2 loaded\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get test images (use rotated images for testing)
    test_dir = Path("data/m2_rotated/val/images")
    if not test_dir.exists():
        test_dir = Path("data/m2_upright_gt/val/images")

    test_images = list(test_dir.glob("*.jpg"))[:num_samples]

    if not test_images:
        print(f"No images found in {test_dir}")
        print("Trying data/m2_crops...")
        test_dir = Path("data/m2_crops")
        test_images = list(test_dir.glob("*.jpg"))[:num_samples]

    print(f"Testing on {len(test_images)} images from {test_dir}")
    print("="*60)

    results = []

    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Processing: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  XX Failed to load")
            continue

        print(f"  Original size: {img.shape[1]}x{img.shape[0]}")

        # Align with M2
        result = aligner.align_with_info(img)

        aligned = result['aligned_image']
        detected_angle = result['detected_angle']
        correction_angle = result['correction_angle']

        print(f"  Detected angle: {detected_angle:.2f}°")
        print(f"  Correction: {correction_angle:.2f}°")
        print(f"  Aligned size: {aligned.shape[1]}x{aligned.shape[0]}")

        # Save results
        base_name = img_path.stem

        # Original
        orig_path = output_path / f"{base_name}_1_original.jpg"
        cv2.imwrite(str(orig_path), img)

        # Aligned
        aligned_path = output_path / f"{base_name}_2_aligned.jpg"
        cv2.imwrite(str(aligned_path), aligned)

        # Create side-by-side comparison
        h, w = img.shape[:2]

        # Resize to same height for comparison
        target_h = 480
        img_resized = cv2.resize(img, (int(w * target_h / h), target_h))

        aligned_h, aligned_w = aligned.shape[:2]
        aligned_resized = cv2.resize(aligned, (int(aligned_w * target_h / aligned_h), target_h))

        # Concatenate side by side
        comparison = np.hstack([
            img_resized,
            aligned_resized
        ])

        # Add labels
        cv2.putText(comparison, "ORIGINAL", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Angle: {detected_angle:.1f}°", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(comparison, "ALIGNED", (img_resized.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, f"Corrected: {correction_angle:.1f}°",
                   (img_resized.shape[1] + 10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        comp_path = output_path / f"{base_name}_3_comparison.jpg"
        cv2.imwrite(str(comp_path), comparison)

        print(f"  >> Saved to: {output_path}")
        print(f"     - {base_name}_1_original.jpg")
        print(f"     - {base_name}_2_aligned.jpg")
        print(f"     - {base_name}_3_comparison.jpg")

        results.append({
            'image': img_path.name,
            'detected_angle': detected_angle,
            'correction': correction_angle,
            'original_size': (img.shape[1], img.shape[0]),
            'aligned_size': (aligned.shape[1], aligned.shape[0]),
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total processed: {len(results)}")

    if results:
        avg_detected = np.mean([r['detected_angle'] for r in results])
        abs_corrections = [abs(r['correction']) for r in results]
        avg_correction = np.mean(abs_corrections)
        max_correction = max(abs_corrections)

        print(f"Avg detected angle: {avg_detected:.2f}°")
        print(f"Avg correction needed: {avg_correction:.2f}°")
        print(f"Max correction: {max_correction:.2f}°")

        # Count how many needed significant correction (> 5°)
        significant = sum(1 for c in abs_corrections if c > 5)
        print(f"Images needing >5° correction: {significant}/{len(results)}")

    print(f"\n>> All results saved to: {output_path.absolute()}")
    print("\nYou can view the comparison images to verify alignment quality.")


def test_m2_interactive(
    model_path: str = "model/orientation.pth",
    image_path: str = None
):
    """
    Interactive test with matplotlib visualization.

    Args:
        model_path: Path to M2 model
        image_path: Path to single test image
    """
    import matplotlib.pyplot as plt

    # Initialize M2
    print("Loading M2 model...")
    aligner = M2Inference(model_path, device='cpu')
    print(">> M2 loaded\n")

    # Get image
    if image_path is None:
        # Use first image from val set
        test_dir = Path("data/m2_rotated/val/images")
        if not test_dir.exists():
            test_dir = Path("data/m2_upright_gt/val/images")

        images = list(test_dir.glob("*.jpg"))
        if not images:
            print("No test images found")
            return

        image_path = str(images[0])

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return

    print(f"Testing: {Path(image_path).name}")
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")

    # Align
    result = aligner.align_with_info(img)

    aligned = result['aligned_image']
    detected = result['detected_angle']
    correction = result['correction_angle']

    print(f"Detected angle: {detected:.2f}°")
    print(f"Correction: {correction:.2f}°")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original\n{img.shape[1]}x{img.shape[0]}")
    axes[0].axis('off')

    # Aligned
    axes[1].imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Aligned (Upright)\n{aligned.shape[1]}x{aligned.shape[0]}")
    axes[1].axis('off')

    # Comparison with rotation info
    # Resize to match heights
    h = 400
    img_small = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    aligned_small = cv2.resize(aligned, (int(aligned.shape[1] * h / aligned.shape[0]), h))

    comparison = np.hstack([img_small, aligned_small])

    # Add text overlays
    cv2.putText(comparison, "ORIGINAL", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(comparison, f"Angle: {detected:.1f}°", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(comparison, "ALIGNED", (img_small.shape[1] + 10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(comparison, f"Corrected: {correction:.1f}°",
               (img_small.shape[1] + 10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    axes[2].imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Side-by-Side Comparison")
    axes[2].axis('off')

    plt.suptitle(f"M2 Alignment Test - {Path(image_path).name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print(f"\n>> Displayed visualization")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test M2 alignment visualization")
    parser.add_argument("--model", type=str, default="model/orientation.pth",
                       help="Path to M2 model")
    parser.add_argument("--image", type=str,
                       help="Single image to test (for interactive mode)")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to test")
    parser.add_argument("--output-dir", type=str, default="test_m2_results",
                       help="Output directory for results")
    parser.add_argument("--interactive", action="store_true",
                       help="Show interactive matplotlib visualization")

    args = parser.parse_args()

    if args.interactive:
        test_m2_interactive(args.model, args.image)
    else:
        test_m2_on_samples(
            model_path=args.model,
            num_samples=args.samples,
            output_dir=args.output_dir
        )
