"""
Test M2 + Smart Rotate with NEWLY TRAINED MODEL
Create visualizations for M1_crops from test_pipeline
"""

import sys
import os
sys.path.insert(0, '.')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm


# ==================== MODEL (GroupNorm + Tanh) ====================
class M2_OrientationModel(nn.Module):
    """M2 Orientation Model - Architecture from Colab"""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.angle_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, 1024),
            nn.GroupNorm(32, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
            nn.Tanh()
        )

    def forward(self, x):
        feats = self.backbone(x)
        vec = self.angle_head(feats)
        return vec


# ==================== M2 SMART ROTATOR ====================
class M2_SmartRotator:
    """M2 Orientation + Smart Rotate with CORRECTED logic"""

    def __init__(self, model_path: str):
        print("[M2] Loading NEWLY TRAINED model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = M2_OrientationModel(pretrained=False).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"      Model: {model_path}")
        print(f"      Epoch: {checkpoint.get('epoch')}")
        print(f"      Val Loss: {checkpoint.get('val_loss'):.6f}")
        print(f"      Device: {self.device}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_angle(self, image: np.ndarray) -> float:
        """Predict rotation angle"""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vec = self.model(tensor)

        # vec is (cos, sin) in [-1, 1] from Tanh
        # FIXED: index 0 is COS, index 1 is SIN
        cos_val = vec[0, 0].cpu().item()
        sin_val = vec[0, 1].cpu().item()

        angle_rad = np.arctan2(sin_val, cos_val)
        angle_deg = np.degrees(angle_rad)  # (-180, 180] range - shortest rotation path

        return angle_deg

    def smart_rotate(self, image: np.ndarray, angle: float) -> tuple:
        """
        Smart rotate image - SIMPLIFIED LOGIC

        Logic: Rotate counter-clockwise by detected angle to bring to 0°
        """
        # Correction angle: rotate counter-clockwise to bring to 0
        correction_angle = -angle

        # Rotate
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        rotation_matrix[0, 2] += (new_width - width) // 2
        rotation_matrix[1, 2] += (new_height - height) // 2

        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        return rotated, correction_angle

    def process_and_visualize(self, image: np.ndarray, save_path: Path, name: str):
        """Process image and create visualization"""
        # Predict angle
        detected_angle = self.predict_angle(image)

        # Rotate
        rotated, correction_angle = self.smart_rotate(image, detected_angle)

        # Create visualization
        h, w = image.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Original
        vis[:, :w] = image

        # Rotated
        rotated_resized = cv2.resize(rotated, (w, h))
        vis[:, w:] = rotated_resized

        # Add text
        cv2.putText(vis, f"Detected: {detected_angle:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Correction: {correction_angle:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        # Save
        cv2.imwrite(str(save_path / f"{name}_m2_rotation.jpg"), vis)

        return {
            'detected_angle': detected_angle,
            'correction_angle': correction_angle,
            'rotated_shape': rotated.shape
        }


# ==================== MAIN ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test M2 + Smart Rotate on test_pipeline M1 crops")
    parser.add_argument("--input", type=str,
                       default=r"F:\Workspace\Project\results\test_pipeline\m1_crops",
                       help="Path to M1 crops directory")
    parser.add_argument("--model", type=str,
                       default=r"F:\Workspace\Project\model\m2_angle_model_best (2).pth",
                       help="Path to M2 model")
    parser.add_argument("--output", type=str,
                       default=r"F:\Workspace\Project\results\m2_new_model_test",
                       help="Output directory")
    parser.add_argument("--samples", type=int, default=20,
                       help="Number of samples to test")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)

    print("="*70)
    print("M2 + SMART ROTATE TEST - NEW MODEL")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Samples: {args.samples}")
    print("="*70)
    print()

    # Initialize M2 rotator
    rotator = M2_SmartRotator(args.model)

    # Get M1 crop images
    input_dir = Path(args.input)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if args.samples > 0:
        image_files = image_files[:args.samples]

    if not image_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(image_files)} images...")
    print()

    angles = []

    for i, img_path in enumerate(tqdm(image_files, desc="Processing")):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Process
        name = img_path.stem
        result = rotator.process_and_visualize(img, output_path, name)

        angles.append({
            'filename': img_path.name,
            'detected': result['detected_angle'],
            'correction': result['correction_angle']
        })

    # Print statistics
    if angles:
        detected_angles = [a['detected'] for a in angles]
        correction_angles = [a['correction'] for a in angles]

        print()
        print("="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Total images: {len(angles)}")
        print()
        print(f"Detected Angles:")
        print(f"  Mean:     {np.mean(detected_angles):.2f}°")
        print(f"  Std:      {np.std(detected_angles):.2f}°")
        print(f"  Min:      {np.min(detected_angles):.2f}°")
        print(f"  Max:      {np.max(detected_angles):.2f}°")
        print()
        print(f"Correction Angles:")
        print(f"  Mean:     {np.mean(correction_angles):.2f}°")
        print(f"  Std:      {np.std(correction_angles):.2f}°")
        print(f"  Min:      {np.min(correction_angles):.2f}°")
        print(f"  Max:      {np.max(correction_angles):.2f}°")
        print()
        print(f"Output directory: {output_path.absolute()}")
        print("="*70)
