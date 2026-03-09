"""
Test GroupNorm vs LayerNorm - which one matches metadata?
"""

import sys
sys.path.insert(0, '.')

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from pathlib import Path
import pandas as pd

MODEL_PATH = r"F:\Workspace\Project\model\m2_angle_model_best (2).pth"
M1_CROPS_DIR = r"F:\Workspace\Project\results\test_pipeline\m1_crops"
METADATA_PATH = r"F:\Workspace\Project\results\test_pipeline\m2_aligned\metadata.csv"

# ==================== Architecture 1: GroupNorm ====================
class M2_GroupNorm(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from collections import OrderedDict

        resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        angle_head_layers = OrderedDict([
            ('0', nn.Identity()),
            ('1', nn.Linear(25088, 1024)),
            ('2', nn.GroupNorm(32, 1024)),     # GROUPNORM
            ('3', nn.ReLU()),
            ('4', nn.Dropout(0.3)),
            ('5', nn.Linear(1024, 512)),
            ('6', nn.GroupNorm(16, 512)),      # GROUPNORM
            ('7', nn.ReLU()),
            ('8', nn.Dropout(0.2)),
            ('9', nn.Linear(512, 2)),
        ])
        self.angle_head = nn.Sequential(angle_head_layers)

    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.flatten(start_dim=1)
        vec = self.angle_head(feats)
        return vec  # NO Tanh, NO Normalize


# ==================== Architecture 2: LayerNorm ====================
class M2_LayerNorm(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        from collections import OrderedDict

        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        angle_head_layers = OrderedDict([
            ('0', nn.Identity()),
            ('1', nn.Linear(25088, 1024)),
            ('2', nn.LayerNorm(1024)),         # LAYERNORM
            ('3', nn.ReLU()),
            ('4', nn.Dropout(dropout)),
            ('5', nn.Linear(1024, 512)),
            ('6', nn.LayerNorm(512)),          # LAYERNORM
            ('7', nn.ReLU()),
            ('8', nn.Dropout(dropout)),
            ('9', nn.Linear(512, 2)),
        ])
        self.angle_head = nn.Sequential(angle_head_layers)

    def forward(self, x):
        features = self.backbone(x)
        features = features.flatten(start_dim=1)
        vec = self.angle_head(features)
        return vec  # NO Normalize


def test_architecture(model_class, model_path, m1_dir, metadata_df, arch_name):
    """Test architecture against metadata"""
    print(f"\n{'='*70}")
    print(f"Testing: {arch_name}")
    print(f"{'='*70}")

    # Load model
    model = model_class()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded: Missing={len(missing)}, Unexpected={len(unexpected)}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test on first 5 images from metadata
    metadata_files = metadata_df['filename'].tolist()[:5]
    m1_dir = Path(m1_dir)

    results = []
    for fname in metadata_files:
        img_path = m1_dir / fname
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            vec = model(tensor)

        cos_val = vec[0, 0].item()  # Index 0 is COS
        sin_val = vec[0, 1].item()  # Index 1 is SIN
        angle_rad = np.arctan2(sin_val, cos_val)  # arctan2(sin, cos)
        angle_deg = np.degrees(angle_rad) % 360

        # Get metadata angle
        meta_row = metadata_df[metadata_df['filename'] == fname]
        if len(meta_row) > 0:
            meta_angle = meta_row.iloc[0]['angle']

            # Calculate angular difference correctly (considering 359° and 1° are close)
            diff = abs(angle_deg - meta_angle)
            angular_diff = min(diff, 360 - diff)

            results.append({
                'filename': fname,
                'predicted': angle_deg,
                'metadata': meta_angle,
                'diff': angular_diff
            })

    if not results:
        print("  No results!")
        return None

    # Print results
    print(f"\n  Predictions vs Metadata:")
    for r in results:
        print(f"    {r['filename'][:40]}...")
        print(f"      Predicted: {r['predicted']:.2f}, Metadata: {r['metadata']:.2f}, Diff: {r['diff']:.2f}")

    avg_diff = np.mean([r['diff'] for r in results])
    print(f"\n  Average difference: {avg_diff:.2f} degrees")

    return avg_diff


if __name__ == "__main__":
    print("="*70)
    print("GROUPNORM vs LAYERNORM TEST")
    print("="*70)
    print("Which architecture was used to train m2_angle_model_best (2).pth?")
    print("="*70)

    # Load metadata
    metadata_df = pd.read_csv(METADATA_PATH)

    # Test both architectures
    diff_gn = test_architecture(
        M2_GroupNorm, MODEL_PATH, M1_CROPS_DIR, metadata_df,
        "GroupNorm (from Colab)"
    )

    diff_ln = test_architecture(
        M2_LayerNorm, MODEL_PATH, M1_CROPS_DIR, metadata_df,
        "LayerNorm (current src/m2_orientation/model.py)"
    )

    # Compare
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")

    if diff_gn is not None and diff_ln is not None:
        print(f"\nGroupNorm average error: {diff_gn:.2f}")
        print(f"LayerNorm average error: {diff_ln:.2f}")

        if abs(diff_gn - diff_ln) < 1:
            print(f"\n[~] Both architectures give SIMILAR results")
            print(f"    (Difference: {abs(diff_gn - diff_ln):.2f})")
        elif diff_gn < diff_ln:
            print(f"\n[+] GroupNorm is BETTER (lower error)")
            print(f"    Model was likely trained with GroupNorm")
        else:
            print(f"\n[+] LayerNorm is BETTER (lower error)")
            print(f"    Model was likely trained with LayerNorm")
