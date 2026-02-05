"""
M2: Orientation Alignment Module

This module implements sin/cos angle regression for normalizing watermeter
image orientation.

Components:
- M2AngleRegressor: CNN model for sin/cos prediction
- M2Inference: Inference API for prediction and alignment
- config: Configuration constants
- utils: Helper functions

Author: ML Team
Version: 1.0.0
"""

from .model import (
    M2AngleRegressor,
    SinCosLoss,
    angle_to_sin_cos,
    sin_cos_to_angle,
    compute_circular_mae,
)

# Legacy alias for backward compatibility
compute_mae = compute_circular_mae

try:
    from .inference import M2Inference, M2AlignmentError
except ImportError:
    # Inference module may not exist yet
    M2Inference = None
    M2AlignmentError = None

try:
    from .config import M2_CONFIG
except ImportError:
    M2_CONFIG = None

__all__ = [
    "M2AngleRegressor",
    "SinCosLoss",
    "angle_to_sin_cos",
    "sin_cos_to_angle",
    "compute_circular_mae",
    "compute_mae",  # Legacy alias
    "M2Inference",
    "M2AlignmentError",
    "M2_CONFIG",
]

__version__ = "1.0.0"
