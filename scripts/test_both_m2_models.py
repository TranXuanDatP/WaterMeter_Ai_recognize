"""
Test both M2 model architectures to see which one matches the checkpoint predictions
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==================== MODEL 1: Backbone + Angle Head (25088→1024→512→2) ====================
class M2_Model1(nn.Module):
    """Model I created: Backbone + angle_head with OrderedDict"""
    def __init__(self, dropout=0.4):
        super().__init__()
        from torchvision.models import resnet18
        from collections import OrderedDict

        resnet = resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        angle_head_layers = OrderedDict([
            ('0', nn.Identity()),
            ('1', nn.Linear(25088, 1024)),
            ('2', nn.LayerNorm(1024)),
            ('3', nn.ReLU()),
            ('4', nn.Dropout(dropout)),
            ('5', nn.Linear(1024, 512)),
            ('6', nn.LayerNorm(512)),
            ('7', nn.ReLU()),
            ('8', nn.Dropout(dropout)),
            ('9', nn.Linear(512, 2)),
        ])
        self.angle_head = nn.Sequential(angle_head_layers)

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(start_dim=1)
        sin_cos = self.angle_head(features)
        return F.normalize(sin_cos, p=2, dim=1)


# ==================== MODEL 2: Backbone + Attention + Regressor (512→256→128→2) ====================
class M2_Model2(nn.Module):
    """Model from test_pipeline_images_4digit.py"""
    def __init__(self, dropout=0.4):
        super().__init__()
        from torchvision.models import resnet18

        resnet = resnet18(weights=None)
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


def load_and_predict(model_class, model_path, image_path, model_name):
    """Load model and predict angle"""
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")

    # Load model
    model = model_class(dropout=0.4)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Try loading with strict=False first
    try:
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded with strict=False")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            print(f"    Examples: {missing[:5]}")
        print(f"  Unexpected keys: {len(unexpected)}")
    except Exception as e:
        print(f"ERROR loading: {e}")
        return None

    model.eval()

    # Load and transform image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor = transform(pil_img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        sin_cos = model(tensor)

    sin_val = sin_cos[0, 0].item()
    cos_val = sin_cos[0, 1].item()
    angle_rad = np.arctan2(sin_val, cos_val)
    angle_deg = np.degrees(angle_rad)
    angle_deg = (angle_deg + 360) % 360

    print(f"\nPrediction:")
    print(f"  Sin: {sin_val:.4f}, Cos: {cos_val:.4f}")
    print(f"  Angle: {angle_deg:.2f} degrees")

    return angle_deg


if __name__ == "__main__":
    MODEL_PATH = r"F:\Workspace\Project\model\M2_Orientation.pth"
    TEST_IMAGE = r"F:\Workspace\Project\data\images_4digit\meter4_00000_0001e09f7ad5442a832f7b5efb74bf2c.jpg"

    print("="*70)
    print("COMPARING TWO M2 MODEL ARCHITECTURES")
    print("="*70)
    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Test Image: {TEST_IMAGE}")
    print(f"Expected (old metadata): ~23.12 degrees (for different file)")
    print(f"Current predictions: ~82-90 degrees")

    # Test Model 1
    angle1 = load_and_predict(M2_Model1, MODEL_PATH, TEST_IMAGE, "Model 1: Backbone + Angle Head")

    # Test Model 2
    angle2 = load_and_predict(M2_Model2, MODEL_PATH, TEST_IMAGE, "Model 2: Backbone + Attention + Regressor")

    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"Model 1 angle: {angle1:.2f}°")
    print(f"Model 2 angle: {angle2:.2f}°")
    print(f"Difference: {abs(angle1 - angle2):.2f}°")

    if angle1 is not None and angle2 is not None:
        if abs(angle1 - angle2) < 1.0:
            print("\n✓ Both models predict similar angles!")
        else:
            print("\n✗ Models predict DIFFERENT angles!")
            print("  One of these architectures is correct for the checkpoint")
