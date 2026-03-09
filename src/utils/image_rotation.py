"""
Smart Image Rotation Utilities

Provides functions for rotating images with smart canvas expansion
to prevent clipping/corner issues.

Author: Claude Code
Created: 2026-03-08
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def smart_rotate(
    image: np.ndarray,
    angle: float,
    border_mode: int = cv2.BORDER_REPLICATE,
    interpolation: int = cv2.INTER_CUBIC,
    expand_canvas: bool = True,
    crop_to_original: bool = False,
    crop_padding: int = 0
) -> np.ndarray:
    """
    Rotate image with smart cropping to prevent clipping

    When an image is rotated, the corners can get clipped if the canvas
    size isn't adjusted. This function automatically calculates the
    required canvas size to fit the entire rotated image.

    Args:
        image: Input image (BGR or grayscale)
        angle: Rotation angle in degrees
            - Positive = counter-clockwise rotation
            - Negative = clockwise rotation
        border_mode: Pixel extrapolation method
            - cv2.BORDER_REPLICATE: Replicate edge pixels (default)
            - cv2.BORDER_CONSTANT: Fill with constant value
            - cv2.BORDER_REFLECT: Mirror reflection
            - cv2.BORDER_WRAP: Wrap around
        interpolation: Interpolation method
            - cv2.INTER_CUBIC: High quality (default, slow)
            - cv2.INTER_LINEAR: Medium quality (fast)
            - cv2.INTER_NEAREST: Fast, no interpolation
        expand_canvas: If True, expand canvas to fit rotated image
                     If False, keep original size (may clip corners)
        crop_to_original: If True, crop back to original size after rotation
                         (useful to maintain image dimensions)
        crop_padding: Padding to add when cropping back to original
                     (positive = larger crop, negative = tighter crop)

    Returns:
        Rotated image (size depends on expand_canvas and crop_to_original)

    Example:
        >>> img = cv2.imread('meter.jpg')
        >>> # Rotate 23.1 degrees clockwise, keep original size
        >>> aligned = smart_rotate(img, -23.1, crop_to_original=True)
        >>> cv2.imwrite('aligned.jpg', aligned)
    """
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle)

    if not expand_canvas:
        # Simple rotation without canvas expansion
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=interpolation,
            borderMode=border_mode
        )
        return rotated

    # Calculate new canvas size to fit rotated image
    # The rotated image's bounding box has different dimensions
    new_w = int(h * abs(np.sin(angle_rad)) + w * abs(np.cos(angle_rad)))
    new_h = int(h * abs(np.cos(angle_rad)) + w * abs(np.sin(angle_rad)))

    # Create rotation matrix
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Adjust translation to center the rotated image on new canvas
    M[0, 2] += (new_w - w) // 2
    M[1, 2] += (new_h - h) // 2

    # Perform rotation
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=interpolation,
        borderMode=border_mode
    )

    # Crop back to original size if requested
    if crop_to_original:
        # Calculate crop coordinates (center crop)
        start_x = (rotated.shape[1] - w) // 2
        start_y = (rotated.shape[0] - h) // 2

        # Add padding
        start_x = max(0, start_x - crop_padding)
        start_y = max(0, start_y - crop_padding)
        end_x = min(rotated.shape[1], start_x + w + 2 * crop_padding)
        end_y = min(rotated.shape[0], start_y + h + 2 * crop_padding)

        # Recalculate start positions if end is at boundary
        if end_x == rotated.shape[1]:
            start_x = max(0, end_x - w - 2 * crop_padding)
        if end_y == rotated.shape[0]:
            start_y = max(0, end_y - h - 2 * crop_padding)

        # Perform crop
        cropped = rotated[start_y:end_y, start_x:end_x]

        # If crop is smaller than original, pad it
        if cropped.shape[0] < h or cropped.shape[1] < w:
            # Pad to original size
            pad_top = max(0, (h - cropped.shape[0]) // 2)
            pad_bottom = max(0, h - cropped.shape[0] - pad_top)
            pad_left = max(0, (w - cropped.shape[1]) // 2)
            pad_right = max(0, w - cropped.shape[1] - pad_left)

            if pad_top + pad_bottom + pad_left + pad_right > 0:
                cropped = cv2.copyMakeBorder(
                    cropped, pad_top, pad_bottom, pad_left, pad_right,
                    border_mode
                )

        return cropped

    return rotated


def rotate_with_crop(
    image: np.ndarray,
    angle: float,
    crop_to_original: bool = True,
    border_value: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Rotate image and optionally crop back to original size

    This is useful when you want to maintain the original image dimensions
    after rotation, accepting that some corners may be clipped.

    Args:
        image: Input image (BGR or grayscale)
        angle: Rotation angle in degrees
        crop_to_original: If True, crop back to original size after rotation
        border_value: Fill color for border (B, G, R) if using BORDER_CONSTANT

    Returns:
        Rotated image (same size as input if crop_to_original=True)

    Example:
        >>> img = cv2.imread('meter.jpg')
        >>> # Rotate and keep original size
        >>> aligned = rotate_with_crop(img, -15.0, crop_to_original=True)
    """
    # First, rotate with expanded canvas
    border_mode = cv2.BORDER_CONSTANT if border_value is not None else cv2.BORDER_REPLICATE
    rotated = smart_rotate(image, angle, border_mode=border_mode)

    if not crop_to_original:
        return rotated

    # Crop back to original size (center crop)
    h, w = image.shape[:2]
    rh, rw = rotated.shape[:2]

    # Calculate crop coordinates (center the original image)
    start_x = (rw - w) // 2
    start_y = (rh - h) // 2

    # Handle case where rotated image is smaller than original
    if start_x < 0 or start_y < 0:
        # Pad with border value
        pad_x = max(-start_x, 0)
        pad_y = max(-start_y, 0)
        rotated = cv2.copyMakeBorder(rotated, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        start_x = (rotated.shape[1] - w) // 2
        start_y = (rotated.shape[0] - h) // 2

    # Crop to original size
    cropped = rotated[start_y:start_y + h, start_x:start_x + w]

    return cropped


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-180, 180] range for minimal rotation

    Args:
        angle: Input angle in degrees

    Returns:
        Normalized angle in range [-180, 180]

    Example:
        >>> normalize_angle(350)  # Returns -10
        >>> normalize_angle(200)  # Returns -160
        >>> normalize_angle(-200) # Returns 160
    """
    # Normalize to [-180, 180]
    if angle <= -180:
        angle += 360
    elif angle > 180:
        angle -= 360
    return angle


def get_minimal_rotation_angle(current_angle: float, target_angle: float = 0.0) -> float:
    """
    Calculate minimal rotation angle needed to reach target

    Args:
        current_angle: Current orientation in degrees [0, 360)
        target_angle: Target orientation in degrees (default: 0.0 = upright)

    Returns:
        Minimal rotation angle in range [-180, 180]
        - Positive = rotate counter-clockwise
        - Negative = rotate clockwise

    Example:
        >>> # If image is at 350°, only need -10° rotation to reach 0°
        >>> get_minimal_rotation_angle(350, 0)  # Returns -10.0
        >>>
        >>> # If image is at 10°, need -10° rotation to reach 0°
        >>> get_minimal_rotation_angle(10, 0)   # Returns -10.0
    """
    # Calculate difference
    diff = target_angle - current_angle

    # Normalize to [-180, 180]
    return normalize_angle(diff)


def auto_rotate(
    image: np.ndarray,
    current_angle: float,
    target_angle: float = 0.0,
    crop_to_size: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Automatically rotate image from current angle to target angle

    Args:
        image: Input image
        current_angle: Current orientation in degrees
        target_angle: Target orientation (default: 0.0 = upright)
        crop_to_size: If True, crop back to original size after rotation
                      (recommended for meter images to maintain focus)
        **kwargs: Additional arguments passed to smart_rotate()

    Returns:
        Rotated image aligned to target angle

    Example:
        >>> img = cv2.imread('meter.jpg')
        >>> # Image is currently rotated 23.1°, align to upright
        >>> aligned = auto_rotate(img, 23.1, target_angle=0.0)
    """
    # Calculate minimal rotation needed
    correction_angle = get_minimal_rotation_angle(current_angle, target_angle)

    # Skip if very small rotation
    if abs(correction_angle) < 1.0:
        return image

    # Apply rotation with crop_to_original if crop_to_size is True
    return smart_rotate(image, correction_angle, crop_to_original=crop_to_size, **kwargs)


# ============================================
# DEMO / TESTING
# ============================================

if __name__ == "__main__":
    print("Smart Rotation Utilities")
    print("=" * 60)

    # Test angle normalization
    print("\nAngle Normalization Tests:")
    test_angles = [350, 356.2, 345.9, 10, 23.1, -10, 200]
    for angle in test_angles:
        normalized = normalize_angle(angle)
        print(f"  {angle:7.1f}° -> {normalized:7.1f}°")

    # Test minimal rotation calculation
    print("\nMinimal Rotation Tests:")
    test_cases = [350, 356.2, 345.9, 10, 23.1, 1.2]
    for angle in test_cases:
        correction = get_minimal_rotation_angle(angle, 0)
        print(f"  At {angle:6.1f}° -> rotate by {correction:6.1f}°")

    print("\n" + "=" * 60)
    print("[OK] All utility functions available for import!")
    print("\nUsage:")
    print("  from src.utils.image_rotation import smart_rotate")
    print("  aligned = smart_rotate(img, -23.1)")
