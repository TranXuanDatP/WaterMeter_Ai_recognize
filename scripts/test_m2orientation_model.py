"""
Test m2orientation.pth to check if it matches:
1. Architecture (GroupNorm + Tanh)
2. Metadata predictions from test_pipeline/m2_aligned
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
import pandas as pd


# ==================== MODEL ARCHITECTURES ====================

# Architecture 1: GroupNorm + Tanh (CORRECT from Colab)
class M2_GroupNorm_Tanh(nn.Module):
    """M2 with GroupNorm + Tanh - Correct architecture"""
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


# Architecture 2: LayerNorm + Normalize (WRONG - current src/m2_orientation/model.py)
class M2_LayerNorm_Normalize(nn.Module):
    """M2 with LayerNorm + Normalize - Wrong architecture"""
    def __init__(self, dropout=0.4):
        super().__init__()
        from collections import OrderedDict

        resnet = models.resnet18(weights=None)
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


def check_model_architecture(model_path):
    """Check which architecture matches the checkpoint"""
    print("="*70)
    print("CHECKING MODEL ARCHITECTURE")
    print("="*70)

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Get checkpoint keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Total keys: {len(state_dict)}")

    # Check key patterns
    print(f"\nKey patterns:")
    key_patterns = {}
    for key in state_dict.keys():
        if 'angle_head' in key:
            parts = key.split('.')
            if len(parts) > 1:
                layer_num = parts[1]
                layer_type = parts[2] if len(parts) > 2 else 'unknown'
                key_patterns[layer_num] = layer_type

    for num in sorted(key_patterns.keys()):
        print(f"  angle_head.{num}: {key_patterns[num]}")

    # Detect architecture
    has_grouprnorm = any('GroupNorm' in k for k in state_dict.keys())
    has_layernorm = any('LayerNorm' in k for k in state_dict.keys())

    print(f"\nArchitecture detection:")
    print(f"  Has GroupNorm: {has_grouprnorm}")
    print(f"  Has LayerNorm: {has_layernorm}")

    if has_grouprnorm:
        print(f"\n  [+] This is GROUPNORM + TANH architecture (CORRECT)")
        return 'grouprnorm_tanh'
    elif has_layernorm:
        print(f"\n  [-] This is LAYERNORM + NORMALIZE architecture (WRONG)")
        return 'layernorm_normalize'
    else:
        print(f"\n  [?] Unknown architecture")
        return 'unknown'


def load_and_predict(model_class, model_path, image_tensor, model_name):
    """Load model and predict"""
    try:
        model = model_class()
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if len(missing) > 10 or len(unexpected) > 10:
            return None

        model.eval()
        with torch.no_grad():
            vec = model(image_tensor)

        return vec
    except:
        return None


def compare_with_metadata(model_path, m1_crops_dir, metadata_path):
    """Compare model predictions with metadata"""
    print("\n" + "="*70)
    print("COMPARING WITH METADATA")
    print("="*70)

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)

    # Get first 5 images
    m1_dir = Path(m1_crops_dir)
    metadata_files = metadata_df['filename'].tolist()[:5]

    # Prepare transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load model
    print(f"\nLoading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Try both architectures
    print("\nTrying Architecture 1: GroupNorm + Tanh")
    model1 = M2_GroupNorm_Tanh(pretrained=False)
    try:
        missing1, unexpected1 = model1.load_state_dict(state_dict, strict=False)
        print(f"  Missing: {len(missing1)}, Unexpected: {len(unexpected1)}")

        if len(missing1) < 5 and len(unexpected1) < 5:
            print("  [+] GroupNorm + Tanh architecture matches!")

            model1.eval()
            results1 = []

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
                    vec = model1(tensor)

                sin_val = vec[0, 0].item()
                cos_val = vec[0, 1].item()
                angle_rad = np.arctan2(sin_val, cos_val)
                angle_deg = np.degrees(angle_rad) % 360

                # Get metadata angle
                meta_row = metadata_df[metadata_df['filename'] == fname]
                if len(meta_row) > 0:
                    meta_angle = meta_row.iloc[0]['angle']

                    results1.append({
                        'filename': fname,
                        'predicted': angle_deg,
                        'metadata': meta_angle,
                        'diff': abs(angle_deg - meta_angle)
                    })

            if results1:
                print("\n  Predictions vs Metadata:")
                for r in results1:
                    print(f"    {r['filename'][:40]}...")
                    print(f"      Predicted: {r['predicted']:.2f}°, Metadata: {r['metadata']:.2f}°, Diff: {r['diff']:.2f}°")

                avg_diff = np.mean([r['diff'] for r in results1])
                print(f"\n  Average difference: {avg_diff:.2f}°")

                if avg_diff < 5:
                    print("  [+] EXCELLENT MATCH! This model produces the metadata results!")
                elif avg_diff < 30:
                    print("  [~] Partial match - similar but not exact")
                else:
                    print("  [-] Poor match - predictions differ significantly")

    except Exception as e:
        print(f"  [-] Failed to load: {e}")


if __name__ == "__main__":
    MODEL_PATH = r"F:\Workspace\Project\model\m2orientation.pth"
    M1_CROPS_DIR = r"F:\Workspace\Project\results\test_pipeline\m1_crops"
    METADATA_PATH = r"F:\Workspace\Project\results\test_pipeline\m2_aligned\metadata.csv"

    print("="*70)
    print("M2ORIENTATION.PTH ARCHITECTURE AND METADATA TEST")
    print("="*70)
    print(f"Model: {MODEL_PATH}")
    print(f"M1 Crops: {M1_CROPS_DIR}")
    print(f"Metadata: {METADATA_PATH}")
    print("="*70)

    # Step 1: Check architecture
    arch_type = check_model_architecture(MODEL_PATH)

    # Step 2: Compare with metadata
    compare_with_metadata(MODEL_PATH, M1_CROPS_DIR, METADATA_PATH)

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
