"""
Backup pipeline results and create comprehensive report
"""
import shutil
from pathlib import Path
from datetime import datetime
import json

# Paths
PIPELINE_DIR = Path(r"F:\Workspace\Project\results\pipeline_full_m1_m2_m3_m3_5_m4_beam")
BACKUP_BASE = Path(r"F:\Workspace\Project\results\backups")

# Create backup with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = BACKUP_BASE / f"pipeline_beam_search_{timestamp}"
backup_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PIPELINE BACKUP")
print("="*70)
print(f"\nBacking up to: {backup_dir}")

# Copy all results
items_to_backup = [
    "pipeline_results.csv",
    "stats.json",
    "m1_crops",
    "m2_aligned",
    "m3_roi_crops",
    "m3_5_black_digits"
]

for item in items_to_backup:
    src = PIPELINE_DIR / item
    dst = backup_dir / item

    if src.exists():
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"  [FILE] {item}")
        else:
            shutil.copytree(src, dst)
            print(f"  [DIR]  {item}")

# Create backup manifest
manifest = {
    'timestamp': timestamp,
    'backup_location': str(backup_dir),
    'original_location': str(PIPELINE_DIR),
    'items_backed_up': items_to_backup,
    'pipeline_config': {
        'm1_confidence': 0.15,
        'm3_confidence': 0.10,
        'beam_width': 10,
        'm2_model': 'm2_angle_model_epoch15_FIXED_COS_SIN.pth',
        'decoder': 'beam_search'
    }
}

with open(backup_dir / "backup_manifest.json", 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\nBackup complete!")
print(f"Location: {backup_dir}")
print("="*70)
