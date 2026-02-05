"""
M2 Inference Module - Orientation Alignment

Based on technical report: Bao_Cao_Ky_Thuat_M2
Results: MAE ~1.57°

Usage:
    aligner = M2Inference("model/orientation.pth")
    aligned_img, angle = aligner.align(cropped_image)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

try:
    from PIL import Image
    from torchvision import models, transforms
except ImportError as e:
    raise ImportError(f"Required dependencies not installed: {e}")

logger = logging.getLogger(__name__)


class M2AlignmentError(Exception):
    """M2 Alignment Exception"""
    pass


def smart_rotate(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Smart rotation algorithm to prevent clipping.

    Calculates new bounding box size to ensure entire image is preserved
    after rotation without cropping corners.

    Math:
        W_new = |H * sin(θ)| + |W * cos(θ)|
        H_new = |H * cos(θ)| + |W * sin(θ)|

    Args:
        image: Input image (numpy array)
        angle: Rotation angle in degrees (counter-clockwise)

    Returns:
        Rotated image with expanded canvas
    """
    h, w = image.shape[:2]

    if angle == 0:
        return image

    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding box dimensions
    cos_val = np.abs(M[0, 0])
    sin_val = np.abs(M[0, 1])

    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    # Adjust translation to center the image
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform rotation with INTER_CUBIC for better quality
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


class M2Inference:
    """
    M2 Orientation Alignment Inference API

    Aligns watermeter images to upright orientation using trained ResNet18 model.

    Architecture (from Bao_Cao_Ky_Thuat_M2):
        - Backbone: ResNet18 (pretrained on ImageNet)
        - Head: FC(512 -> 256 -> 2) with dropout (0.2)
        - Output: [sin(θ), cos(θ)] normalized to unit circle
        - Loss: MSE on sin/cos components
        - Angle decoding: atan2(sin, cos) → θ

    Performance:
        - MAE: ~1.57°
        - Input: 224x224 RGB images
        - Smart rotation to prevent clipping

    Example:
        >>> aligner = M2Inference("model/orientation.pth")
        >>> aligned_img, angle = aligner.align(cropped_image)
        >>> print(f"Detected angle: {angle:.2f}°")
        >>> cv2.imwrite("aligned.jpg", aligned_img)
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        input_size: int = 224
    ):
        """
        Initialize M2 inference.

        Args:
            model_path: Path to trained model weights (.pth file)
            device: 'cpu' or 'cuda'
            input_size: Model input size (default: 224)
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.model_path = Path(model_path)

        logger.info(f"Initializing M2 Inference:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Input size: {input_size}x{input_size}")

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Setup transforms (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        logger.info("✅ M2 Inference initialized successfully")

    def _load_model(self) -> torch.nn.Module:
        """
        Load ResNet18 model with custom regression head.

        Architecture MUST match training (saved with wrapper class):
            - backbone: ResNet18
            - head: FC(512 -> 256) + ReLU + Dropout(0.2) + FC(256 -> 2)

        Returns:
            Loaded PyTorch model
        """
        # Define model architecture matching training (from Bao_Cao_Ky_Thuat_M2)
        # The model was saved with a wrapper class containing:
        #   - backbone: ResNet18 object (with layer1, layer2, layer3, layer4, etc.)
        #   - head: Sequential regression head

        class WaterMeterAlignerModel(nn.Module):
            """Exact model architecture from training"""
            def __init__(self):
                super().__init__()
                # Backbone: ResNet18 with FC layer removed
                self.backbone = models.resnet18(weights=None)
                self.backbone.fc = nn.Identity()  # Remove original FC layer

                # Head: Custom regression
                self.head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(256, 2)  # Output: [sin, cos]
                )

            def forward(self, x):
                # Extract features (backbone already has avgpool + flatten via fc=Identity)
                features = self.backbone(x)
                # Apply regression head
                output = self.head(features)
                return output

        model = WaterMeterAlignerModel()

        # Load weights
        if not self.model_path.exists():
            raise M2AlignmentError(f"Model file not found: {self.model_path}")

        try:
            state_dict = torch.load(
                str(self.model_path),
                map_location=self.device,
                weights_only=False
            )
            model.load_state_dict(state_dict)
            logger.info(f"✅ Loaded model weights from {self.model_path}")
        except Exception as e:
            raise M2AlignmentError(f"Failed to load model: {e}")

        model.to(self.device)
        return model

    def predict_angle(self, image: np.ndarray) -> float:
        """
        Predict rotation angle from input image.

        Args:
            image: Input image (BGR numpy array from cv2)

        Returns:
            Rotation angle in degrees [0, 360)

        Process:
            1. Convert BGR → RGB → PIL
            2. Apply transforms (resize to 224x224, normalize)
            3. Forward pass through model
            4. Decode [sin, cos] → angle using atan2
        """
        # Convert CV2 (BGR) to PIL (RGB)
        if len(image.shape) == 2:
            # Grayscale → RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(image_rgb)

        # Transform
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(tensor)  # Shape: [1, 2]
            sin_cos = output.cpu().numpy()[0]  # [sin, cos]

        # Decode angle using atan2 (handles all 4 quadrants correctly)
        sin_val, cos_val = sin_cos[0], sin_cos[1]
        angle_rad = np.arctan2(sin_val, cos_val)
        angle_deg = np.degrees(angle_rad)

        # Normalize to [0, 360)
        angle = (angle_deg + 360) % 360

        return angle

    def align(
        self,
        image: np.ndarray,
        return_angle: bool = True
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Align image to upright orientation using smart rotation.

        Args:
            image: Input cropped watermeter image (BGR numpy array)
            return_angle: If True, return (aligned_image, angle)
                        If False, return aligned_image only

        Returns:
            Aligned image (upright, 0°), and optionally the detected angle

        Example:
            >>> aligned, angle = aligner.align(cropped_img)
            >>> print(f"Corrected {angle:.2f}° rotation")
            >>> cv2.imwrite("upright.jpg", aligned)
        """
        # Predict current angle
        current_angle = self.predict_angle(image)

        # Calculate correction angle (rotate in reverse)
        correction_angle = -current_angle

        # Apply smart rotation (prevents clipping)
        aligned_image = smart_rotate(image, correction_angle)

        if return_angle:
            return aligned_image, current_angle
        else:
            return aligned_image

    def align_with_info(self, image: np.ndarray) -> Dict[str, any]:
        """
        Align image and return detailed information.

        Args:
            image: Input cropped watermeter image (BGR numpy array)

        Returns:
            Dictionary with:
                - aligned_image: Aligned image (BGR numpy array)
                - detected_angle: Detected rotation angle [0, 360)
                - correction_angle: Angle applied to correct [-360, 360)
                - input_shape: Input image shape
                - output_shape: Output image shape
        """
        # Predict current angle
        detected_angle = self.predict_angle(image)
        correction_angle = -detected_angle

        # Apply smart rotation
        aligned_image = smart_rotate(image, correction_angle)

        return {
            "aligned_image": aligned_image,
            "detected_angle": detected_angle,
            "correction_angle": correction_angle,
            "input_shape": image.shape,
            "output_shape": aligned_image.shape
        }

    def align_batch(self, images: list) -> Tuple[list, list]:
        """
        Align multiple images.

        Args:
            images: List of input images (BGR numpy arrays)

        Returns:
            Tuple of (aligned_images, angles)
        """
        aligned_images = []
        angles = []

        for img in images:
            aligned, angle = self.align(img)
            aligned_images.append(aligned)
            angles.append(angle)

        return aligned_images, angles


# ==========================================
# CLI Usage
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="M2 Orientation Alignment Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image alignment
  python inference.py --model model/orientation.pth --input crop.jpg --output aligned.jpg

  # Batch alignment
  python inference.py --model model/orientation.pth --input crop.jpg --output aligned.jpg --device cpu

Based on: Bao_Cao_Ky_Thuat_M2 (MAE: ~1.57°)
        """
    )

    parser.add_argument("--model", type=str, default="model/orientation.pth",
                       help="Path to trained model (default: model/orientation.pth)")
    parser.add_argument("--input", type=str, required=True,
                       help="Input image path")
    parser.add_argument("--output", type=str, default="aligned_output.jpg",
                       help="Output image path (default: aligned_output.jpg)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed information")

    args = parser.parse_args()

    # Initialize aligner
    try:
        aligner = M2Inference(args.model, device=args.device)
    except M2AlignmentError as e:
        print(f"❌ Error initializing aligner: {e}")
        exit(1)

    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ Error: Cannot load image from {args.input}")
        exit(1)

    # Align with detailed info
    print(f"📷 Processing: {args.input}")
    result = aligner.align_with_info(image)

    # Save result
    cv2.imwrite(args.output, result["aligned_image"])

    # Print results
    print(f"✅ Alignment complete!")
    print(f"   Input shape: {result['input_shape']}")
    print(f"   Detected angle: {result['detected_angle']:.2f}°")
    print(f"   Correction: {result['correction_angle']:.2f}°")
    print(f"   Output shape: {result['output_shape']}")
    print(f"   Output: {args.output}")

    if args.verbose:
        print(f"\n📊 Detailed Information:")
        print(f"   MAE (from training): ~1.57°")
        print(f"   Model: ResNet18 with sin/cos regression")
        print(f"   Smart rotation: Enabled (prevents clipping)")
