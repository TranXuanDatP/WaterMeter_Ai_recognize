"""
M4 Inference Module

CRNN model inference for meter reading
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict

from .model import CRNN, CTCDecoder


class M4Inference:
    """
    M4 CRNN Inference Class

    Handles loading trained CRNN model and performing inference
    on digit box regions from M3.
    """

    def __init__(self, model_path: str, device: str = "cpu", num_chars: int = 11):
        """
        Initialize M4 inference

        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference ('cpu' or 'cuda')
            num_chars: Number of output classes (default: 11)

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        # Check model file exists
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model_path = model_path
        self.device = torch.device(device)
        self.num_chars = num_chars

        # Load model
        self.model = CRNN(num_chars=num_chars, hidden_size=256, num_layers=2)
        self.load_model(model_path)

        # Initialize decoder
        self.decoder = CTCDecoder(num_chars=num_chars)

        # Model parameters
        self.img_height = 64
        self.img_width = 160

    def load_model(self, model_path: Path):
        """
        Load trained model checkpoint

        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded from {model_path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for CRNN

        Args:
            image: Input image (grayscale or color)

        Returns:
            Preprocessed tensor (1, 1, H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to fixed height, maintain aspect ratio for width
        h, w = image.shape
        scale = self.img_height / h
        new_w = int(w * scale)

        resized = cv2.resize(image, (new_w, self.img_height))

        # Pad to fixed width if needed
        if new_w < self.img_width:
            padded = cv2.copyMakeBorder(
                resized,
                0, 0,  # top, bottom
                0, self.img_width - new_w,  # left, right
                cv2.BORDER_CONSTANT,
                value=255  # White padding
            )
        else:
            # Crop if too wide
            padded = resized[:, :self.img_width]

        # Normalize to [-1, 1]
        normalized = padded.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5

        # Convert to tensor
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

        return tensor

    def predict(self, image: np.ndarray) -> str:
        """
        Predict digit sequence from image

        Args:
            image: Input image (from M3 digit box)

        Returns:
            Digit sequence (e.g., "12345")
        """
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)

        # Decode
        text = self.decoder.decode(logits)

        return text

    def predict_with_info(self, image: np.ndarray) -> Dict:
        """
        Predict with detailed information

        Args:
            image: Input image

        Returns:
            Dictionary with:
                - text: Digit sequence
                - confidence: Average confidence score
                - probs: Per-digit probability distributions
        """
        # Preprocess
        input_tensor = self.preprocess(image).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(input_tensor)

        # Get probabilities
        probs = torch.softmax(logits, dim=2)  # (T, 1, C)

        # Decode
        text = self.decoder.decode(logits)

        # Calculate confidence (average max probability)
        max_probs = probs.max(dim=2)[0]  # (T, 1)
        confidence = max_probs.mean().item()

        return {
            'text': text,
            'confidence': confidence,
            'logits': logits.cpu().numpy()
        }

    def predict_batch(self, images: list) -> list:
        """
        Predict on batch of images

        Args:
            images: List of input images

        Returns:
            List of predicted digit sequences
        """
        results = []
        for image in images:
            text = self.predict(image)
            results.append(text)
        return results


if __name__ == "__main__":
    # Test inference (requires trained model)
    print("M4 CRNN Inference Module")
    print("This module requires a trained model checkpoint.")
    print("\nUsage:")
    print("  from src.m4_crnn_reading.inference import M4Inference")
    print("  model = M4Inference('path/to/model.pth')")
    print("  result = model.predict(image)")
