"""
M2 Module Configuration

Configuration constants for sin/cos angle regression model.
"""

from dataclasses import dataclass


@dataclass
class M2ModelConfig:
    """Sin/Cos Angle Regressor Configuration."""

    # Model architecture
    backbone: str = "resnet18"  # ResNet-18 or mobilenet_v2
    pretrained: bool = True  # Use ImageNet pretrained weights
    dropout: float = 0.3  # Dropout rate for regularization

    # Input/Output
    input_size: int = 640  # Input image size (square)
    output_dim: int = 2  # sin and cos outputs

    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005

    # Learning rate scheduler
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    patience: int = 10  # Stop if no improvement for 10 epochs
    min_delta: float = 0.1  # Minimum change in MAE (degrees)

    # Data augmentation
    rotation_range: int = 30  # Rotation augmentation (degrees)
    flip_prob: float = 0.5  # Horizontal flip probability

    # Performance targets
    target_mae: float = 5.0  # Target MAE in degrees
    target_inference_time_ms: float = 15.0  # <15ms on GPU

    # Paths
    model_save_path: str = "checkpoints/orientation/"
    checkpoint_interval: int = 5  # Save checkpoint every N epochs

    # MLflow tracking
    mlflow_enabled: bool = True
    mlflow_experiment_name: str = "M2_Orientation_Alignment"

    # Hardware
    device: str = "cuda"  # cuda or cpu
    num_workers: int = 4
    pin_memory: bool = True


# Default configuration instance
M2_CONFIG = M2ModelConfig()


def get_config() -> M2ModelConfig:
    """Get default M2 configuration."""
    return M2_CONFIG
