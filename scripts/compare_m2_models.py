"""
Compare all M2 model files to find which one matches the architecture:
- Backbone: ResNet18 (without final FC layer)
- Regression Head: 512 features → (cos, sin) output
- Training: MSE Loss on (cos, sin) vectors
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

# ==================== CORRECT ARCHITECTURE FROM USER'S DESCRIPTION ====================
class M2_Orientation_Correct(nn.Module):
    """
    M2 Orientation Model - Correct architecture from user's specification

    Backbone: ResNet18 (bỏ lớp phân loại cuối)
    Regression Head:
      - Input: 512*7*7 = 25088 features from ResNet18
      - Hidden: 1024 → 512
      - Output: (cos α, sin α) in [-1, 1] from Tanh
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # ResNet18 backbone (bỏ lớp phân loại cuối)
        resnet = models.resnet18(weights='DEFAULT' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 512, 7, 7)

        # Regression Head
        self.angle_head = nn.Sequential(
            nn.Flatten(),                    # 512*7*7 = 25088
            nn.Linear(512*7*7, 1024),        # 25088 → 1024
            nn.GroupNorm(32, 1024),          # GroupNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),            # 1024 → 512
            nn.GroupNorm(16, 512),           # GroupNorm
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),               # 512 → 2 (cos, sin)
            nn.Tanh()                        # Output in [-1, 1]
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - RGB image
        Returns:
            vec: (B, 2) - (cos α, sin α) in [-1, 1]
        """
        feats = self.backbone(x)  # (B, 512, 7, 7)
        vec = self.angle_head(feats)  # (B, 2)
        return vec  # NO normalization! Return raw Tanh output


def test_model_loading(model_path, model_class, model_name):
    """Test if model can be loaded with the given architecture"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")

    if not Path(model_path).exists():
        print("  [!] File not found!")
        return None

    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Get checkpoint info
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', 'N/A')
        train_loss = checkpoint.get('train_loss', 'N/A')
        best_val_loss = checkpoint.get('best_val_loss', 'N/A')

        print(f"  Checkpoint Info:")
        print(f"    Epoch: {epoch}")
        print(f"    Val Loss: {val_loss}")
        print(f"    Train Loss: {train_loss}")
        print(f"    Best Val Loss: {best_val_loss}")

        # Get state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        print(f"  Total keys in state_dict: {len(state_dict)}")

        # Try to load with correct architecture
        print(f"\n  Loading with M2_Orientation_Correct architecture...")
        model = model_class(pretrained=False)

        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            print(f"    Missing keys: {len(missing)}")
            if missing:
                print(f"      Examples: {missing[:3]}")

            print(f"    Unexpected keys: {len(unexpected)}")
            if unexpected:
                print(f"      Examples: {unexpected[:3]}")

            # Check if keys match
            if len(missing) == 0 and len(unexpected) == 0:
                print(f"    [+] PERFECT MATCH! Architecture is correct!")
                return model, checkpoint, 'perfect'
            elif len(missing) < 5 and len(unexpected) < 5:
                print(f"    [~] GOOD MATCH (mostly compatible)")
                return model, checkpoint, 'good'
            else:
                print(f"    [-] POOR MATCH (architecture mismatch)")
                return None, checkpoint, 'poor'

        except Exception as e:
            print(f"    [-] ERROR: {e}")
            return None, checkpoint, 'error'

    except Exception as e:
        print(f"  [-] ERROR loading checkpoint: {e}")
        return None


def check_architecture_details(state_dict):
    """Analyze state_dict to understand architecture"""
    print(f"\n  Architecture Analysis:")

    # Check for backbone
    has_backbone = any('backbone' in k for k in state_dict.keys())
    has_resnet = any('conv1' in k or 'layer1' in k or 'layer2' in k for k in state_dict.keys())

    # Check for angle_head
    has_angle_head = any('angle_head' in k for k in state_dict.keys())

    # Check for normalization types
    has_grouprnorm = any('GroupNorm' in k or 'gn' in k.lower() for k in state_dict.keys())
    has_layernorm = any('LayerNorm' in k or 'ln' in k.lower() for k in state_dict.keys())
    has_batchnorm = any('BatchNorm' in k or 'bn' in k.lower() for k in state_dict.keys())

    # Check for output activation
    has_tanh = any('tanh' in k.lower() for k in state_dict.keys())
    has_sigmoid = any('sigmoid' in k.lower() for k in state_dict.keys())

    print(f"    Has backbone: {has_backbone or has_resnet}")
    print(f"    Has angle_head: {has_angle_head}")
    print(f"    Has GroupNorm: {has_grouprnorm}")
    print(f"    Has LayerNorm: {has_layernorm}")
    print(f"    Has BatchNorm: {has_batchnorm}")
    print(f"    Has Tanh: {has_tanh}")
    print(f"    Has Sigmoid: {has_sigmoid}")

    # Show some key samples
    print(f"\n  Sample keys:")
    keys_list = list(state_dict.keys())[:10]
    for key in keys_list:
        print(f"    {key}")


if __name__ == "__main__":
    MODEL_DIR = Path(r"F:\Workspace\Project\model")

    # List all potential M2 model files
    model_files = [
        ("M2_Orientation.pth", "Original M2 Orientation"),
        ("m2_angle_model_best (2).pth", "Newly trained (Epoch 15)"),
        ("m2_angle_model_last.pth_epoch18.pth", "Epoch 18 checkpoint"),
        ("m2orientation.pth", "Named 'm2orientation' (actually OCR?)"),
    ]

    print("="*70)
    print("M2 MODEL ARCHITECTURE VERIFICATION")
    print("="*70)
    print("Correct Architecture:")
    print("  Backbone: ResNet18 (without final FC)")
    print("  Head: 25088 -> 1024 -> 512 -> 2")
    print("  Output: (cos, sin) in [-1, 1] via Tanh")
    print("  Loss: MSE on (cos, sin) vectors")
    print("="*70)

    results = {}

    for filename, description in model_files:
        model_path = MODEL_DIR / filename
        result = test_model_loading(str(model_path), M2_Orientation_Correct, description)

        if result is None or result[0] is None:
            if result is not None and len(result) > 1 and result[1] is not None:
                checkpoint = result[1]
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                check_architecture_details(state_dict)

            results[filename] = {'match_quality': 'failed'}
            continue

        model, checkpoint, match_quality = result

        # Get state_dict for detailed analysis
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        check_architecture_details(state_dict)

        results[filename] = {
            'match_quality': match_quality,
            'epoch': checkpoint.get('epoch', 'N/A'),
            'val_loss': checkpoint.get('val_loss', 'N/A'),
            'best_val_loss': checkpoint.get('best_val_loss', 'N/A')
        }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for filename, description in model_files:
        if filename in results:
            result = results[filename]
            match = result['match_quality']
            epoch = result.get('epoch', 'N/A')

            if match == 'perfect':
                print(f"  [+] {description}")
                print(f"      File: {filename}")
                print(f"      Epoch: {epoch}")
                print(f"      Status: PERFECT MATCH - This is the correct architecture!")
            elif match == 'good':
                print(f"  [~] {description}")
                print(f"      File: {filename}")
                print(f"      Epoch: {epoch}")
                print(f"      Status: Good match")
            elif match == 'poor':
                print(f"  [-] {description}")
                print(f"      File: {filename}")
                print(f"      Status: Poor match - wrong architecture")
            else:
                print(f"  [X] {description}")
                print(f"      File: {filename}")
                print(f"      Status: Failed to load")
        else:
            print(f"  [?] {description}")
            print(f"      File: {filename}")
            print(f"      Status: File not found")

    print("\n" + "="*70)
