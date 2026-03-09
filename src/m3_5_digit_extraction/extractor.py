"""
M3.5 Digit Extractor - Extract black digits from ROI images

Can be used standalone or integrated into pipeline.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class M3_5_DigitExtractor:
    """
    M3.5: Extract black digits from ROI by detecting and removing red digits

    Methods:
        - detect_red_digit_region: Find red digit x-coordinate
        - extract: Crop black digits from ROI
        - extract_from_file: Load and extract from file path
    """

    def __init__(self, min_crop_ratio: float = 0.75, fallback_ratio: float = 0.8):
        """
        Initialize M3.5 extractor

        Args:
            min_crop_ratio: Minimum width ratio to keep (safety check)
            fallback_ratio: Fallback ratio if red detection fails
        """
        self.min_crop_ratio = min_crop_ratio
        self.fallback_ratio = fallback_ratio

    def detect_red_digit_region(self, img: np.ndarray) -> int:
        """
        Detect the red digit region on the right side

        Args:
            img: Input image (BGR)

        Returns:
            x-coordinate where red digits start
        """
        h, w = img.shape[:2]

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define red color range (two ranges because red wraps around in HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return int(w * self.fallback_ratio)

        # Get all red regions
        red_regions = []
        for cnt in contours:
            x, y, w_red, h_red = cv2.boundingRect(cnt)
            if w_red > 5 and h_red > 10:  # Filter small noise
                red_regions.append((x, y, w_red, h_red))

        if len(red_regions) == 0:
            return int(w * self.fallback_ratio)

        # Find the rightmost red region (decimal digits)
        red_regions.sort(key=lambda r: r[0], reverse=True)
        rightmost_red = red_regions[0]

        return rightmost_red[0]

    def extract(self, roi_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Extract black digits from ROI image

        Args:
            roi_image: ROI image from M3 (BGR)

        Returns:
            Tuple of (black_digits_image, metadata_dict)
        """
        if roi_image is None or roi_image.size == 0:
            return None, {'status': 'error', 'error': 'Invalid input image'}

        h, w = roi_image.shape[:2]

        # Detect red digit region
        red_x_start = self.detect_red_digit_region(roi_image)

        # Crop only black digits (left part, exclude red digits)
        crop_x_end = min(red_x_start, w)
        crop_x_end = max(crop_x_end - 5, int(w * self.min_crop_ratio))

        black_digits = roi_image[:, :crop_x_end]

        crop_h, crop_w = black_digits.shape[:2]

        metadata = {
            'status': 'success',
            'original_size': (w, h),
            'crop_size': (crop_w, crop_h),
            'red_x_start': red_x_start,
            'crop_ratio': crop_x_end / w,
            'crop_x_end': crop_x_end
        }

        return black_digits, metadata

    def extract_from_file(
        self,
        img_path: Path,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Load and extract black digits from file

        Args:
            img_path: Path to ROI image
            output_dir: Optional directory to save cropped image

        Returns:
            Result dictionary with status and metadata
        """
        img = cv2.imread(str(img_path))

        if img is None:
            return {
                'filename': img_path.name,
                'status': 'error',
                'error': 'Could not read image'
            }

        black_digits, metadata = self.extract(img)

        if metadata['status'] == 'error':
            return {
                'filename': img_path.name,
                **metadata
            }

        # Save if output directory is specified
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / img_path.name
            cv2.imwrite(str(output_path), black_digits)
            metadata['output_path'] = str(output_path)

        return {
            'filename': img_path.name,
            **metadata
        }


# Standalone test
if __name__ == "__main__":
    import sys

    # Test paths
    test_image = Path(r"F:\Workspace\Project\results\test_pipeline\m3_roi_crops\meter4_00000_validate_00000_00385501ab4d419fa7b0bdf0d9f8451f_roi.jpg")
    output_dir = Path(r"F:\Workspace\Project\results\test_pipeline\m3_5_test")

    print("M3.5 Digit Extractor Test")
    print("=" * 60)

    if not test_image.exists():
        print(f"Test image not found: {test_image}")
        sys.exit(1)

    # Create extractor
    extractor = M3_5_DigitExtractor()

    # Extract from file
    result = extractor.extract_from_file(test_image, output_dir)

    print(f"\nFilename: {result['filename']}")
    print(f"Status: {result['status']}")

    if result['status'] == 'success':
        print(f"Original size: {result['original_size']}")
        print(f"Crop size: {result['crop_size']}")
        print(f"Crop ratio: {result['crop_ratio']:.3f}")
        print(f"Red X start: {result['red_x_start']}")
        print(f"\nOutput saved to: {result['output_path']}")
    else:
        print(f"Error: {result.get('error', 'Unknown')}")

    print("\n" + "=" * 60)
    print("[OK] M3.5 Extractor ready for integration!")
