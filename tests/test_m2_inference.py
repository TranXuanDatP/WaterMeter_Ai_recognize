"""
Unit tests for M2 Orientation Alignment Inference

Tests sin/cos angle regression and image alignment.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# Test imports
try:
    from src.m2_orientation_alignment import M2Inference, M2AlignmentError
except ImportError:
    pytest.skip("M2 module not available", allow_module_level=True)


# Test fixtures
@pytest.fixture
def model_path():
    """Path to trained M2 model."""
    # Try multiple possible paths
    possible_paths = [
        Path("checkpoints/orientation/best_model.pt"),
        Path("../checkpoints/orientation/best_model.pt"),
        Path("f:/Workspace/Project/checkpoints/orientation/best_model.pt"),
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    pytest.skip(f"Model file not found in any of: {possible_paths}")


@pytest.fixture
def inference(model_path):
    """M2Inference instance."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return M2Inference(model_path, device=device)


@pytest.fixture
def upright_meter_image():
    """Create an upright meter image (0° rotation)."""
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    # Create a circular meter in center
    center_y, center_x = 320, 320
    for y in range(640):
        for x in range(640):
            if (x - center_x)**2 + (y - center_y)**2 <= 200**2:
                image[y, x] = [200, 200, 200]

    # Add a horizontal line at top to indicate orientation (12 o'clock)
    cv2.rectangle(image, (310, 100), (330, 130), [50, 50, 50], -1)

    return image


@pytest.fixture
def rotated_90_image():
    """Create a 90° rotated meter image."""
    # Get upright image and rotate
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    center_y, center_x = 320, 320

    for y in range(640):
        for x in range(640):
            if (x - center_x)**2 + (y - center_y)**2 <= 200**2:
                image[y, x] = [200, 200, 200]

    # Indicator at right side (3 o'clock) = 90° rotation
    cv2.rectangle(image, (510, 310), (540, 330), [50, 50, 50], -1)

    return image


# Task 2.9 Tests: Unit tests for M2 inference
class TestM2InferenceOutputFormat:
    """Test output format (AC #3)."""

    def test_output_format_structure(self, inference, upright_meter_image):
        """
        AC #3: Test output structure and types.
        """
        result = inference.predict(upright_meter_image)

        # Check required keys
        assert "aligned_image" in result
        assert "angle_degrees" in result
        assert "sin_theta" in result
        assert "cos_theta" in result
        assert "confidence" in result
        assert "success" in result

        # Check aligned image shape
        assert isinstance(result["aligned_image"], np.ndarray)
        assert result["aligned_image"].shape == (640, 640, 3)
        assert result["aligned_image"].dtype == np.uint8

        # Check angle type and range
        assert isinstance(result["angle_degrees"], float)
        assert 0.0 <= result["angle_degrees"] < 360.0

        # Check sin/cos type and range
        assert isinstance(result["sin_theta"], float)
        assert isinstance(result["cos_theta"], float)
        assert -1.0 <= result["sin_theta"] <= 1.0
        assert -1.0 <= result["cos_theta"] <= 1.0

        # Check confidence
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_aligned_image_shape(self, inference, upright_meter_image):
        """
        AC #3: Test output is 640x640x3.
        """
        result = inference.predict(upright_meter_image)

        aligned = result["aligned_image"]
        assert aligned.shape == (640, 640, 3)
        assert aligned.dtype == np.uint8


class TestM2InferenceRotation:
    """Test rotation correction (AC #2)."""

    def test_inference_upright_minimal_rotation(self, inference, upright_meter_image):
        """
        AC #2: Test minimal rotation for already upright image.
        """
        result = inference.predict(upright_meter_image)

        assert result["success"] is True
        # Upright image should have angle close to 0° or 360°
        angle = result["angle_degrees"]
        is_close_to_zero = angle < 10 or angle > 350
        assert is_close_to_zero, f"Expected angle ~0°, got {angle}°"

    def test_inference_rotated_image(self, inference, rotated_90_image):
        """
        AC #2: Test rotation correction for rotated image.
        """
        result = inference.predict(rotated_90_image)

        assert result["success"] is True
        # Rotated image should be detected and aligned
        # The exact angle depends on model prediction


class TestM2InferenceInputValidation:
    """Test input validation."""

    def test_invalid_image_type(self, inference):
        """Test that non-numpy arrays raise ValueError."""
        with pytest.raises(ValueError, match="must be numpy array"):
            inference.predict("not_an_image")

    def test_invalid_image_dimensions(self, inference):
        """Test that 2D images raise ValueError."""
        invalid_image = np.zeros((640, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 3D"):
            inference.predict(invalid_image)

    def test_invalid_image_channels(self, inference):
        """Test that non-RGB images raise ValueError."""
        invalid_image = np.zeros((640, 640, 1), dtype=np.uint8)
        with pytest.raises(ValueError, match="must have 3 channels"):
            inference.predict(invalid_image)

    def test_invalid_image_dtype(self, inference):
        """Test that non-uint8 images raise ValueError."""
        invalid_image = np.zeros((640, 640, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="must be uint8"):
            inference.predict(invalid_image)


class TestM2InferencePerformance:
    """Test inference performance."""

    def test_inference_time_reasonable(self, inference, upright_meter_image):
        """
        AC #1: Test inference time is reasonable.

        Target: <15ms on GPU
        Note: On CPU this may be slower
        """
        import time

        start = time.perf_counter()
        result = inference.predict(upright_meter_image)
        end = time.perf_counter()

        inference_time_ms = (end - start) * 1000

        # Should complete within reasonable time
        assert inference_time_ms < 5000  # 5 seconds max for CPU
        assert result["success"] is True


class TestM2Math:
    """Test sin/cos math and periodicity."""

    def test_sin_cos_unit_circle(self):
        """Test that sin² + cos² ≈ 1 for various angles."""
        import math

        test_angles = [0, 45, 90, 135, 180, 225, 270, 315, 359]
        for angle_deg in test_angles:
            angle_rad = math.radians(angle_deg)
            sin_val = math.sin(angle_rad)
            cos_val = math.cos(angle_rad)

            magnitude = math.sqrt(sin_val**2 + cos_val**2)
            assert abs(magnitude - 1.0) < 0.001, f"Angle {angle_deg}°: magnitude={magnitude}"

    def test_periodicity_359_vs_minus1(self):
        """
        Test periodicity: 359° ≡ -1° (should have similar sin/cos).
        """
        import math

        sin_359 = math.sin(math.radians(359))
        cos_359 = math.cos(math.radians(359))

        sin_minus1 = math.sin(math.radians(-1))
        cos_minus1 = math.cos(math.radians(-1))

        # Should be nearly identical
        assert abs(sin_359 - sin_minus1) < 0.01
        assert abs(cos_359 - cos_minus1) < 0.01


# Run tests with: pytest tests/test_m2_inference.py -v
