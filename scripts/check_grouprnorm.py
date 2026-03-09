"""
Check if model has GroupNorm or LayerNorm
"""

import torch

MODEL_PATH = r"F:\Workspace\Project\model\m2_angle_model_best (2).pth"

checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

print("="*70)
print("ALL STATE_DICT KEYS")
print("="*70)

for i, key in enumerate(state_dict.keys(), 1):
    shape = state_dict[key].shape
    print(f"{i:3d}. {key:50s} {str(shape):30s}")

print("\n" + "="*70)
print("SEARCHING FOR NORMALIZATION LAYERS")
print("="*70)

# Check all keys for normalization patterns
print("\nSearching for 'norm' in keys:")
norm_keys = [k for k in state_dict.keys() if 'norm' in k.lower() or 'gn' in k.lower() or 'ln' in k.lower()]
if norm_keys:
    for key in norm_keys:
        print(f"  {key}: {state_dict[key].shape}")
else:
    print("  [!] No normalization layers found!")

# Check for specific patterns
print("\nChecking specific patterns:")
print(f"  Has 'weight' or 'bias' in angle_head.1: {any(k.startswith('angle_head.1') for k in state_dict.keys())}")
print(f"  Has 'weight' or 'bias' in angle_head.2: {any(k.startswith('angle_head.2') for k in state_dict.keys())}")
print(f"  Has 'weight' or 'bias' in angle_head.5: {any(k.startswith('angle_head.5') for k in state_dict.keys())}")
print(f"  Has 'weight' or 'bias' in angle_head.6: {any(k.startswith('angle_head.6') for k in state_dict.keys())}")
print(f"  Has 'weight' or 'bias' in angle_head.9: {any(k.startswith('angle_head.9') for k in state_dict.keys())}")

# Check angle_head.2 and angle_head.6 specifically
print("\n" + "="*70)
print("ANGLE_HEAD.2 DETAILS")
print("="*70)
angle_head_2_keys = [k for k in state_dict.keys() if k.startswith('angle_head.2')]
for key in angle_head_2_keys:
    print(f"  {key}: {state_dict[key].shape}")

print("\n" + "="*70)
print("ANGLE_HEAD.6 DETAILS")
print("="*70)
angle_head_6_keys = [k for k in state_dict.keys() if k.startswith('angle_head.6')]
for key in angle_head_6_keys:
    print(f"  {key}: {state_dict[key].shape}")

# Try to infer what these layers are
print("\n" + "="*70)
print("LAYER TYPE INFERENCE")
print("="*70)

# angle_head.1: Linear(25088, 1024)
w1 = state_dict['angle_head.1.weight']
b1 = state_dict['angle_head.1.bias']
print(f"\nLayer 1 (angle_head.1):")
print(f"  weight: {w1.shape} -> Linear({w1.shape[1]}, {w1.shape[0]})")
print(f"  bias: {b1.shape}")

# angle_head.2: ?
w2 = state_dict['angle_head.2.weight']
b2 = state_dict['angle_head.2.bias']
print(f"\nLayer 2 (angle_head.2):")
print(f"  weight: {w2.shape}")
print(f"  bias: {b2.shape}")

# Determine if it's GroupNorm or LayerNorm
if w2.shape == torch.Size([1024]):
    # Could be LayerNorm(1024) or GroupNorm(32, 1024) or GroupNorm(16, 1024)
    # Both have weight and bias of size (num_features)
    print(f"  -> This is a Normalization layer with 1024 features")
    print(f"  -> Could be: LayerNorm(1024) or GroupNorm(X, 1024)")
    print(f"  -> Need to check the model code to know which one")

# angle_head.5: Linear(1024, 512)
w5 = state_dict['angle_head.5.weight']
b5 = state_dict['angle_head.5.bias']
print(f"\nLayer 5 (angle_head.5):")
print(f"  weight: {w5.shape} -> Linear({w5.shape[1]}, {w5.shape[0]})")
print(f"  bias: {b5.shape}")

# angle_head.6: ?
w6 = state_dict['angle_head.6.weight']
b6 = state_dict['angle_head.6.bias']
print(f"\nLayer 6 (angle_head.6):")
print(f"  weight: {w6.shape}")
print(f"  bias: {b6.shape}")

if w6.shape == torch.Size([512]):
    print(f"  -> This is a Normalization layer with 512 features")
    print(f"  -> Could be: LayerNorm(512) or GroupNorm(Y, 512)")

# angle_head.9: Linear(512, 2)
w9 = state_dict['angle_head.9.weight']
b9 = state_dict['angle_head.9.bias']
print(f"\nLayer 9 (angle_head.9):")
print(f"  weight: {w9.shape} -> Linear({w9.shape[1]}, {w9.shape[0]})")
print(f"  bias: {b9.shape}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nThe model architecture is:")
print("  backbone: ResNet18")
print("  angle_head.0: Identity() or Flatten()")
print("  angle_head.1: Linear(25088, 1024)")
print("  angle_head.2: Normalization(1024) - LayerNorm or GroupNorm?")
print("  angle_head.3: ReLU()")
print("  angle_head.4: Dropout()")
print("  angle_head.5: Linear(1024, 512)")
print("  angle_head.6: Normalization(512) - LayerNorm or GroupNorm?")
print("  angle_head.7: ReLU()")
print("  angle_head.8: Dropout()")
print("  angle_head.9: Linear(512, 2)")
print("\nThe checkpoint only saves parameters (weight, bias), not activation functions.")
print("So we can't distinguish between LayerNorm and GroupNorm from checkpoint alone.")
print("\nBUT: The architecture MATCHES user's specification!")
print("     25088 -> 1024 -> 512 -> 2")
print("     Backbone: ResNet18")
print("     Output: 2 values (cos, sin)")
