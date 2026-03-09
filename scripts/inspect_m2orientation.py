"""
Inspect m2orientation.pth checkpoint structure in detail
"""
import torch
import torchvision.models as models

MODEL_PATH = r"F:\Workspace\Project\model\m2orientation.pth"

print("="*70)
print("INSPECTING M2ORIENTATION.PTH")
print("="*70)

checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\nModel info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A')}")
else:
    state_dict = checkpoint

print(f"\nTotal parameters: {len(state_dict)}")
print("\nAll state_dict keys:")
for i, key in enumerate(state_dict.keys(), 1):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"  {i:2d}. {key:60s} {shape}")

# Analyze structure
print("\n" + "="*70)
print("STRUCTURE ANALYSIS")
print("="*70)

backbone_keys = [k for k in state_dict.keys() if 'backbone' in k]
angle_head_keys = [k for k in state_dict.keys() if 'angle_head' in k]

print(f"\nBackbone keys: {len(backbone_keys)}")
for key in backbone_keys[:10]:
    print(f"  {key}")

print(f"\nAngle head keys: {len(angle_head_keys)}")
for key in angle_head_keys:
    print(f"  {key}")

# Check what type of normalization
print("\n" + "="*70)
print("NORMALIZATION LAYERS")
print("="*70)

has_bn = any('bn' in k.lower() or 'batch_norm' in k.lower() for k in state_dict.keys())
has_gn = any('gn' in k.lower() or 'group_norm' in k.lower() for k in state_dict.keys())
has_ln = any('ln' in k.lower() or 'layer_norm' in k.lower() for k in state_dict.keys())
has_identity = any('identity' in k.lower() for k in state_dict.keys())

print(f"Has BatchNorm: {has_bn}")
print(f"Has GroupNorm: {has_gn}")
print(f"Has LayerNorm: {has_ln}")
print(f"Has Identity: {has_identity}")

# Analyze angle_head structure
print("\n" + "="*70)
print("ANGLE_HEAD STRUCTURE")
print("="*70)

# Group by layer index
angle_head_layers = {}
for key in angle_head_keys:
    parts = key.split('.')
    if len(parts) >= 2:
        layer_idx = parts[1]
        if layer_idx.isdigit():
            if layer_idx not in angle_head_layers:
                angle_head_layers[layer_idx] = []
            angle_head_layers[layer_idx].append(key)

for idx in sorted(angle_head_layers.keys()):
    print(f"\nLayer {idx}:")
    for key in angle_head_layers[idx]:
        print(f"  {key}")
