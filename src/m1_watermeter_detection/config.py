"""
M1 Module Configuration

Configuration constants for YOLOv8 watermeter detection model.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class M1ModelConfig:
    """YOLOv8 model configuration."""

    # Model architecture
    model_name: str = "yolov8s.pt"  # Small variant (11M parameters)
    input_size: int = 640  # Input image size (square)
    num_classes: int = 1  # Single class: watermeter

    # Detection thresholds
    confidence_threshold: float = 0.50  # Minimum confidence for detection
    nms_threshold: float = 0.45  # Non-Maximum Suppression IoU threshold

    # Training hyperparameters
    batch_size: int = 16
    epochs: int = 100
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    final_lr_fraction: float = 0.01  # lr0 * lrf = final learning rate

    # Learning rate scheduler
    scheduler: str = "cosine"  # Cosine annealing

    # Early stopping
    patience: int = 10  # Stop if no improvement for 10 epochs
    min_delta: float = 0.001  # Minimum change to qualify as improvement

    # Data augmentation (standard YOLO)
    hsv_h: float = 0.015  # Hue augmentation
    hsv_s: float = 0.7  # Saturation augmentation
    hsv_v: float = 0.4  # Value (brightness) augmentation
    degrees: float = 0.0  # Rotation (+/- degrees)
    translate: float = 0.1  # Translation (fraction of image size)
    scale: float = 0.5  # Scale (gain)
    shear: float = 0.0  # Shear (degrees)
    perspective: float = 0.0  # Perspective transform
    flipud: float = 0.0  # Vertical flip probability
    fliplr: float = 0.5  # Horizontal flip probability
    mosaic: float = 1.0  # Mosaic augmentation probability
    mixup: float = 0.0  # MixUp augmentation probability

    # Advanced augmentation for robustness
    blur_prob: float = 0.1  # Gaussian blur probability
    brightness_prob: float = 0.1  # Random brightness probability
    contrast_prob: float = 0.1  # Random contrast probability

    # Performance targets
    target_map: float = 0.98  # 98% mAP@0.5
    target_inference_time_ms: float = 20.0  # <20ms on GPU

    # Paths
    model_save_path: str = "models/m1/"  # Where to save trained models
    checkpoint_interval: int = 5  # Save checkpoint every N epochs

    # MLflow tracking
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = "M1_Watermeter_Detection"

    # Hardware
    device: str = "cuda"  # cuda or cpu
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # Pin memory for faster GPU transfer


# Default configuration instance
M1_CONFIG = M1ModelConfig()

# YOLO dataset configuration template
DATASET_YAML_TEMPLATE = """# M1 Watermeter Detection Dataset Configuration
path: {dataset_path}
train: {train_path}
val: {val_path}

nc: 1  # number of classes
names: ['watermeter']  # class names
"""


# Class names mapping
CLASS_NAMES = {
    0: "watermeter"
}


# Model card metadata
MODEL_CARD = {
    "version": "1.0.0",
    "architecture": "YOLOv8s",
    "parameters": "11M",
    "input_size": "640x640x3",
    "output": "Cropped meter region (640x640x3) + bounding box",
    "training_data": "1,000 labeled watermeter images",
    "target_map": "98% @ 0.5 IoU",
    "target_inference_time": "<20ms (GPU)",
    "classes": 1,
    "use_case": "Watermeter detection in full images",
}


def get_config(**overrides) -> M1ModelConfig:
    """
    Get configuration with optional overrides.

    Args:
        **overrides: Configuration values to override

    Returns:
        M1ModelConfig instance with applied overrides

    Example:
        config = get_config(batch_size=32, epochs=200)
    """
    config_dict = M1_CONFIG.__dict__.copy()
    config_dict.update(overrides)
    return M1ModelConfig(**config_dict)
