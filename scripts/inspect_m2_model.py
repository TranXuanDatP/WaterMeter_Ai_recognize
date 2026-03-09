"""
Inspect M2 Model without full inference
"""
import torch

checkpoint = torch.load(r"F:\Workspace\Project\model\M2_Orientation.pth", map_location='cpu')

print("=" * 80)
print("M2 MODEL INFO")
print("=" * 80)

print(f"\nEpoch: {checkpoint['epoch']}")
print(f"Validation Loss: {checkpoint['val_loss']:.4f}")

state_dict = checkpoint['model_state_dict']
print(f"\nTotal layers: {len(state_dict)}")

print("\n=== Backbone Structure ===")
backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone')]
for key in sorted(backbone_keys)[:15]:
    print(f"  {key}")

print("\n=== Angle Head Structure ===")
head_keys = [k for k in state_dict.keys() if k.startswith('angle_head')]
for key in sorted(head_keys):
    shape = state_dict[key].shape
    print(f"  {key:40s} {shape}")

print("\n=== Model Size ===")
param_size = sum(p.numel() for p in state_dict.values())
print(f"  Total parameters: {param_size:,}")
print(f"  Model size (MB): {param_size * 4 / 1024 / 1024:.1f}")

print("\n" + "=" * 80)
