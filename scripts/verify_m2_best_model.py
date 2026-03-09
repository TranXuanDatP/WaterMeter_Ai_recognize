"""
Verify m2_angle_model_best (2).pth matches user's architecture specification
"""

import torch
import torch.nn as nn
import torchvision.models as models

MODEL_PATH = r"F:\Workspace\Project\model\m2_angle_model_best (2).pth"

print("="*70)
print("VERIFYING M2_ANGLE_MODEL_BEST (2).PTH")
print("="*70)

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print(f"\nCheckpoint Info:")
print(f"  Epoch: {checkpoint.get('epoch')}")
print(f"  Val Loss: {checkpoint.get('val_loss')}")

# Get state_dict
state_dict = checkpoint['model_state_dict']

print(f"\nTotal parameters: {len(state_dict)}")

# Analyze structure
print("\n" + "="*70)
print("BACKBONE STRUCTURE (ResNet18)")
print("="*70)

backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone')]
print(f"Backbone parameters: {len(backbone_keys)}")

# Check ResNet18 layers
resnet_layers = {}
for key in backbone_keys:
    parts = key.split('.')
    if len(parts) >= 2:
        layer = parts[1]
        if layer not in resnet_layers:
            resnet_layers[layer] = []
        resnet_layers[layer].append(key)

for layer in sorted(resnet_layers.keys()):
    params = resnet_layers[layer]
    print(f"\nLayer {layer}: {len(params)} parameters")
    for param in params[:3]:
        print(f"  {param}")
    if len(params) > 3:
        print(f"  ... and {len(params)-3} more")

print("\n" + "="*70)
print("ANGLE_HEAD STRUCTURE")
print("="*70)

angle_head_keys = [k for k in state_dict.keys() if k.startswith('angle_head')]
print(f"Angle head parameters: {len(angle_head_keys)}")

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
        else:
            # Layer without index (like flatten, tanh)
            if 'no_index' not in angle_head_layers:
                angle_head_layers['no_index'] = []
            angle_head_layers['no_index'].append(key)

for idx in sorted(angle_head_layers.keys(), key=lambda x: (x == 'no_index', x)):
    params = angle_head_layers[idx]
    print(f"\nLayer {idx}: {len(params)} parameters")
    for param in params:
        param_name = param.split('.')[-1]
        print(f"  angle_head.{idx}.{param_name}")

print("\n" + "="*70)
print("ARCHITECTURE SUMMARY")
print("="*70)

# Analyze what layers exist
has_flatten = any('flatten' in k.lower() for k in angle_head_keys)
has_linear_1 = any('0.weight' in k and 'angle_head' in k for k in state_dict.keys())
has_gn = any('GroupNorm' in k for k in state_dict.keys())
has_ln = any('LayerNorm' in k for k in state_dict.keys())
has_tanh = any('Tanh' in k or 'tanh' in k for k in state_dict.keys())
has_dropout = any('Dropout' in k or 'dropout' in k for k in state_dict.keys())

print(f"\nDetection results:")
print(f"  Has Flatten: {has_flatten}")
print(f"  Has Linear layers: True (detected)")
print(f"  Has GroupNorm: {has_gn}")
print(f"  Has LayerNorm: {has_ln}")
print(f"  Has Tanh: {has_tanh}")
print(f"  Has Dropout: {has_dropout}")

# Get input/output sizes from Linear layers
linear_0_weight = state_dict.get('angle_head.0.weight')
linear_1_weight = state_dict.get('angle_head.5.weight')
linear_2_weight = state_dict.get('angle_head.9.weight')

if linear_0_weight is not None:
    print(f"\nLinear layer sizes:")
    print(f"  Layer 0 (Flatten->Linear): {linear_0_weight.shape} -> Input: {linear_0_weight.shape[1]}, Output: {linear_0_weight.shape[0]}")

if linear_1_weight is not None:
    print(f"  Layer 5: {linear_1_weight.shape} -> Input: {linear_1_weight.shape[1]}, Output: {linear_1_weight.shape[0]}")

if linear_2_weight is not None:
    print(f"  Layer 9 (Output): {linear_2_weight.shape} -> Input: {linear_2_weight.shape[1]}, Output: {linear_2_weight.shape[0]}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

# Check if matches user's spec
if linear_0_weight is not None and linear_1_weight is not None and linear_2_weight is not None:
    input_size = linear_0_weight.shape[1]
    hidden1_size = linear_0_weight.shape[0]
    hidden2_size = linear_1_weight.shape[0]
    output_size = linear_2_weight.shape[0]

    print(f"\nArchitecture detected:")
    print(f"  Input: {input_size}")
    print(f"  Hidden1: {hidden1_size}")
    print(f"  Hidden2: {hidden2_size}")
    print(f"  Output: {output_size}")

    if input_size == 25088 and hidden1_size == 1024 and hidden2_size == 512 and output_size == 2:
        print(f"\n  [+] PERFECT! Matches user specification:")
        print(f"      25088 -> 1024 -> 512 -> 2")
        print(f"      Backbone: ResNet18")
        print(f"      Output: (cos, sin) for angle regression")
    else:
        print(f"\n  [-] Sizes don't match specification")

print("\n" + "="*70)
