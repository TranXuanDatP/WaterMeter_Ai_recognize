"""
M1: Watermeter Detection Module

This module implements YOLOv8-based watermeter detection for the
6-module meter reading pipeline.

Components:
- M1Model: Training and model management
- M1Inference: Inference API for prediction
- config: Configuration constants
- utils: Helper functions

Author: ML Team
Version: 1.0.0
"""

from .model import M1Model
from .inference import M1Inference, M1DetectionError
from .config import M1_CONFIG

__all__ = [
    "M1Model",
    "M1Inference",
    "M1DetectionError",
    "M1_CONFIG",
]

__version__ = "1.0.0"
