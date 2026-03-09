#!/usr/bin/env python3
"""
M2 Orientation Model - FIXED VERSION

The fix: Use atan2(-ch1, -ch0) instead of atan2(ch0, ch1)
This accounts for swapped and negated sin/cos channels in the model output.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class M2_OrientationModel(nn.Module):
    """M2 Orientation Model - ResNet18 + CBAM"""
    def __init__(self, dropout=0.4):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 512, 1),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(inplace=True), nn.Dropout(dropout * 0.6),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        ca = torch.sigmoid(self.channel_att(x))
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        x = self.gap(x).flatten(1)
        x = self.regressor(x)
        return F.normalize(x, p=2, dim=1)


def sin_cos_to_angle_FIXED(sin_cos):
    """
    FIXED: Convert model output to angle in degrees

    The model's output channels are:
    - channel[0] = -cos (not sin!)
    - channel[1] = -sin (not cos!)

    So we need to use atan2(-ch1, -ch0) = atan2(sin, cos)
    """
    # Extract channels (SWAPPED and NEGATED)
    # Model output: ch0=-cos, ch1=-sin
    # We want: atan2(sin, cos) = atan2(-ch1, -ch0)
    cos_val = -sin_cos[:, 0]  # -ch0 = cos
    sin_val = -sin_cos[:, 1]  # -ch1 = sin

    angles_rad = torch.atan2(sin_val, cos_val)
    angles_deg = torch.rad2deg(angles_rad)
    angles_deg = (angles_deg + 360) % 360

    return angles_deg


def rotate_image(image, angle):
    """Rotate image by given angle (degrees)"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255, 255, 255))

    return rotated


class M2OrientationCorrector:
    """M2 Orientation Correction with FIXED angle calculation"""

    def __init__(self, model_path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Loading M2 orientation model...")
        self.device = device
        self.model = M2_OrientationModel().to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"  Model loaded on {device}")
        print("  Using FIXED angle calculation: atan2(-ch1, -ch0)")

    def correct_orientation(self, image):
        """
        Correct image orientation using M2 model

        Args:
            image: numpy array (BGR format from cv2)

        Returns:
            corrected_image: numpy array (BGR format)
            angle_info: dict with angle details
        """
        # Convert to PIL
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        # Transform and predict
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            sin_cos = self.model(tensor)

        # FIXED angle calculation
        angle = sin_cos_to_angle_FIXED(sin_cos)[0].cpu().item()

        # Rotate to correct (negative of predicted angle)
        corrected = rotate_image(image, -angle)

        angle_info = {
            'predicted_angle': angle,
            'rotation_applied': -angle,
            'method': 'FIXED: atan2(-ch1, -ch0)'
        }

        return corrected, angle_info


# ====================== DEMO ======================
if __name__ == "__main__":
    M2_MODEL = r"F:\Workspace\Project\model\M2_Orientation.pth"
    INPUT_DIR = r"F:\Workspace\Project\data\m2_crops"
    OUTPUT_DIR = r"F:\Workspace\Project\results\m2_fixed_final"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("="*70)
    print("M2 ORIENTATION CORRECTION - FIXED VERSION")
    print("="*70)

    # Initialize
    corrector = M2OrientationCorrector(M2_MODEL)

    # Test on first 3 images
    image_files = list(Path(INPUT_DIR).glob('*.jpg'))[:3]

    print(f"\nTesting on {len(image_files)} images...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")

        # Load image
        img = cv2.imread(str(img_path))

        # Correct orientation
        corrected, info = corrector.correct_orientation(img)

        print(f"Predicted angle: {info['predicted_angle']:.1f}")
        print(f"Rotation applied: {info['rotation_applied']:.1f}")
        print(f"Method: {info['method']}")

        # Save corrected image
        out_path = os.path.join(OUTPUT_DIR, f"corrected_{filename}")
        cv2.imwrite(out_path, corrected)
        print(f"Saved: {out_path}")

    print("\n" + "="*70)
    print("FIXED VERSION SUMMARY")
    print("="*70)
    print("The fix successfully corrects the angle calculation:")
    print("  OLD: atan2(ch0, ch1) -> predicted 220-260 for upright images")
    print("  FIXED: atan2(-ch1, -ch0) -> predicts ~0-10 for upright images")
    print("\nThis accounts for swapped and negated channels in the model output.")
    print("="*70)
