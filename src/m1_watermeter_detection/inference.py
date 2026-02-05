"""
M1 Inference Implementation

Implements inference API for YOLOv8 watermeter detection.
"""

import logging
from typing import Dict, Tuple, Optional
import time

try:
    import torch
    import numpy as np
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        f"Required dependencies not installed: {e}\n"
        "Please install: pip install torch ultralytics"
    )

from .config import M1_CONFIG, get_config


# Setup logging
logger = logging.getLogger(__name__)


class M1DetectionError(Exception):
    """Raised when watermeter detection fails."""
    pass


class M1Inference:
    """
    YOLOv8 Watermeter Detection Inference

    Provides prediction API for detecting watermeters in full images.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: Optional[float] = None,
        device: Optional[str] = None,
        use_fp16: bool = True,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to trained model (.pt file)
            confidence_threshold: Minimum confidence for detection (default: 0.50)
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
            use_fp16: Use FP16 precision for faster inference (if supported)

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.config = get_config()
        self.confidence_threshold = confidence_threshold or self.config.confidence_threshold
        self.device = device or self.config.device
        self.use_fp16 = use_fp16

        logger.info(f"Loading M1 model from: {model_path}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Device: {self.device}")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)

            # Enable FP16 if requested and supported
            if self.use_fp16 and self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.half()
                logger.info("FP16 precision enabled")

            # Warm-up inference
            self._warmup()

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def _warmup(self):
        """Run warm-up inference to initialize GPU kernels."""
        logger.debug("Running warm-up inference...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = self.model.predict(
                dummy_image,
                verbose=False,
                conf=self.confidence_threshold,
                device=self.device,
            )
            logger.debug("Warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")

    def predict(
        self,
        image: np.ndarray,
        return_all: bool = False,
    ) -> Dict[str, any]:
        """
        Detect watermeter in full image.

        Args:
            image: Full meter image (H, W, 3), uint8, RGB
            return_all: If True, return all detections; else return only highest confidence

        Returns:
            {
                "cropped_region": np.ndarray,  # (640, 640, 3)
                "bounding_box": {
                    "x_min": float,
                    "y_min": float,
                    "x_max": float,
                    "y_max": float,
                    "confidence": float
                },
                "confidence": float,  # 0-1
                "success": bool,
                "inference_time_ms": float,
                "num_detections": int,
            }

        Raises:
            M1DetectionError: If no detection above threshold
            ValueError: If image format is invalid
        """
        # Validate input
        self._validate_input(image)

        # Run inference
        start_time = time.perf_counter()
        results = self.model.predict(
            image,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.config.nms_threshold,
            device=self.device,
            imgsz=self.config.input_size,
        )
        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        # Extract results
        detections = self._parse_results(results, image.shape)

        if not detections:
            logger.warning(f"No watermeter detected above threshold {self.confidence_threshold}")
            raise M1DetectionError(
                f"No watermeter detected. All detections below confidence threshold "
                f"({self.confidence_threshold})"
            )

        # Get best detection (highest confidence)
        if return_all:
            best_detection = detections[0]  # Already sorted by confidence
        else:
            best_detection = detections[0]

        # Crop meter region
        cropped_region = self._crop_meter_region(image, best_detection["bbox"])

        return {
            "cropped_region": cropped_region,
            "bounding_box": best_detection["bbox"],
            "confidence": best_detection["confidence"],
            "success": True,
            "inference_time_ms": inference_time,
            "num_detections": len(detections),
        }

    def _validate_input(self, image: np.ndarray):
        """Validate input image format."""
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Image must be numpy array, got {type(image)}")

        if image.ndim != 3:
            raise ValueError(f"Image must be 3D (H, W, C), got shape {image.shape}")

        if image.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels (RGB), got {image.shape[2]}")

        if image.dtype != np.uint8:
            raise ValueError(f"Image must be uint8, got {image.dtype}")

    def _parse_results(
        self,
        results: list,
        original_shape: Tuple[int, int, int],
    ) -> list:
        """
        Parse YOLO results into detection list.

        Args:
            results: YOLO prediction results
            original_shape: Original image shape (H, W, C)

        Returns:
            List of detections sorted by confidence (descending)
        """
        detections = []

        for result in results:
            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()  # [x_min, y_min, x_max, y_max]
                    conf = float(boxes.conf[i].cpu().numpy())

                    # Convert to normalized coordinates
                    h, w = original_shape[:2]
                    x_min, y_min, x_max, y_max = box

                    detection = {
                        "bbox": {
                            "x_min": float(x_min),
                            "y_min": float(y_min),
                            "x_max": float(x_max),
                            "y_max": float(y_max),
                            "confidence": conf,
                        },
                        "confidence": conf,
                    }
                    detections.append(detection)

        # Sort by confidence (descending)
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return detections

    def _crop_meter_region(
        self,
        image: np.ndarray,
        bbox: Dict[str, float],
    ) -> np.ndarray:
        """
        Crop meter region from image.

        Args:
            image: Full image (H, W, 3)
            bbox: Bounding box dict with x_min, y_min, x_max, y_max

        Returns:
            Cropped region (640, 640, 3)
        """
        x_min = int(bbox["x_min"])
        y_min = int(bbox["y_min"])
        x_max = int(bbox["x_max"])
        y_max = int(bbox["y_max"])

        # Extract region
        cropped = image[y_min:y_max, x_min:x_max]

        # Resize to 640x640 if needed
        if cropped.shape[0] != 640 or cropped.shape[1] != 640:
            import cv2
            cropped = cv2.resize(cropped, (640, 640), interpolation=cv2.INTER_LINEAR)

        return cropped

    def benchmark(
        self,
        images: list,
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            images: List of test images
            num_iterations: Number of inference runs

        Returns:
            Performance metrics (p50, p95, p99 latency in ms)
        """
        logger.info(f"Benchmarking with {len(images)} images, {num_iterations} iterations")

        latencies = []

        for i in range(num_iterations):
            img = images[i % len(images)]
            start = time.perf_counter()
            try:
                self.predict(img)
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms
            except M1DetectionError:
                # Skip failed detections in benchmark
                continue

        if not latencies:
            logger.warning("No successful detections in benchmark")
            return {}

        latencies_array = np.array(latencies)
        metrics = {
            "p50_ms": float(np.percentile(latencies_array, 50)),
            "p95_ms": float(np.percentile(latencies_array, 95)),
            "p99_ms": float(np.percentile(latencies_array, 99)),
            "mean_ms": float(np.mean(latencies_array)),
            "std_ms": float(np.std(latencies_array)),
            "min_ms": float(np.min(latencies_array)),
            "max_ms": float(np.max(latencies_array)),
        }

        logger.info(f"Benchmark results:")
        logger.info(f"  p50: {metrics['p50_ms']:.2f}ms")
        logger.info(f"  p95: {metrics['p95_ms']:.2f}ms")
        logger.info(f"  p99: {metrics['p99_ms']:.2f}ms")

        return metrics
