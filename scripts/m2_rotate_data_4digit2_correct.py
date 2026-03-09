"""
M2 Orientation + Smart Rotate on data_4digit2 (CORRECTED VERSION)

Uses the CORRECT smart rotation logic from the original metadata:
- Normalize angle to [-180, 180]
- Rotate counter-clockwise by angle (correction = -angle)
"""
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.m2_orientation.model import M2_OrientationModel

# Fix encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    # Paths
    DATA_DIR = Path(r"F:\Workspace\Project\data\data_4digit2")
    OUTPUT_DIR = Path(r"F:\Workspace\Project\results\m2_rotated_corrected")
    MODEL_PATH = Path(r"F:\Workspace\Project\model\M2_Orientation.pth")
    LABELS_FILE = Path(r"F:\Workspace\Project\data\images_4digit2.csv")

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # M2 config
    IMG_SIZE = 224

    # Output subdirs
    ALIGNED_DIR = OUTPUT_DIR / "aligned_images"
    LOG_DIR = OUTPUT_DIR / "logs"

# ============================================
# M2 SMART ROTATOR (CORRECTED)
# ============================================

class M2_SmartRotator:
    """M2: Orientation prediction + Smart Rotate (CORRECTED LOGIC)"""

    def __init__(self, model_path: str, device: torch.device):
        print("[M2] Loading orientation + smart rotate model...")
        self.device = device
        self.model = M2_OrientationModel().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        print(f"      Model: {model_path}")
        print(f"      Device: {device}")
        print(f"      Epoch: {checkpoint.get('epoch', 'N/A')}")

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict_angle(self, image: np.ndarray) -> float:
        """
        Predict rotation angle from model

        Returns:
            angle: Angle in degrees [0, 360]
        """
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Transform
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            sin_cos = self.model(tensor)

        # Convert to angle [0, 360]
        sin_val = sin_cos[0, 0].cpu().item()
        cos_val = sin_cos[0, 1].cpu().item()
        angle_rad = np.arctan2(sin_val, cos_val)
        angle_deg = np.degrees(angle_rad)
        angle_deg = (angle_deg + 360) % 360

        return angle_deg

    def smart_rotate(self, image: np.ndarray, angle: float) -> tuple:
        """
        Smart rotate image using CORRECTED logic from metadata

        Logic:
        1. Normalize angle to [-180, 180]
        2. Correction angle = -angle_norm (rotate counter-clockwise)
        3. Rotate by correction angle

        Args:
            image: Input image
            angle: Detected angle in [0, 360]

        Returns:
            (rotated_image, correction_angle_used)
        """
        # Normalize angle to [-180, 180]
        angle_norm = angle if angle <= 180 else angle - 180

        # Correction angle: rotate counter-clockwise to bring to 0
        correction_angle = -angle_norm

        # Rotate
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

        # Calculate new canvas size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))

        # Adjust translation
        rotation_matrix[0, 2] += (new_width - width) // 2
        rotation_matrix[1, 2] += (new_height - height) // 2

        # Rotate
        rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        return rotated, correction_angle

# ============================================
# LOGGER
# ============================================

class RotationLogger:
    """Logger for rotation results"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.start_time = datetime.now()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create log file
        self.log_file = log_dir / f"rotation_run_{self.session_id}.log"

        # Statistics
        self.stats = {
            'total_images': 0,
            'successful': 0,
            'errors': 0,
            'start_time': self.start_time.isoformat(),
            'angles': [],
            'correction_angles': []
        }

    def log(self, message, print_to_console=True):
        """Log message"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        if print_to_console:
            print(message)

    def save_summary(self):
        """Save summary"""
        self.stats['end_time'] = datetime.now().isoformat()
        self.stats['duration_seconds'] = (
            datetime.now() - self.start_time
        ).total_seconds()

        if self.stats['angles']:
            self.stats['angle_stats'] = {
                'mean': float(np.mean(self.stats['angles'])),
                'std': float(np.std(self.stats['angles'])),
                'min': float(np.min(self.stats['angles'])),
                'max': float(np.max(self.stats['angles'])),
                'median': float(np.median(self.stats['angles']))
            }

        if self.stats['correction_angles']:
            self.stats['correction_angle_stats'] = {
                'mean': float(np.mean(self.stats['correction_angles'])),
                'std': float(np.std(self.stats['correction_angles'])),
                'min': float(np.min(self.stats['correction_angles'])),
                'max': float(np.max(self.stats['correction_angles'])),
                'median': float(np.median(self.stats['correction_angles']))
            }

        summary_file = self.log_dir / f"summary_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        return summary_file

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 80)
    print("M2 ORIENTATION + SMART ROTATE ON DATA_4DIGIT2 (CORRECTED VERSION)")
    print("=" * 80)
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Output directory: {Config.OUTPUT_DIR}")
    print(f"Device: {Config.DEVICE}")
    print("=" * 80)

    # Create output directories
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = RotationLogger(Config.LOG_DIR)
    logger.log("=" * 80)
    logger.log("M2 ORIENTATION + SMART ROTATE (CORRECTED VERSION)")
    logger.log("=" * 80)

    # Get image list
    print(f"\n[1/4] Scanning images...")
    image_files = list(Config.DATA_DIR.glob('*.jpg')) + list(Config.DATA_DIR.glob('*.png'))
    print(f"      Found {len(image_files)} images")
    logger.log(f"Found {len(image_files)} images")

    # Load M2 model
    print(f"\n[2/4] Loading M2 model...")
    rotator = M2_SmartRotator(str(Config.MODEL_PATH), Config.DEVICE)
    logger.log("M2 model loaded successfully")

    # Load labels (for reference)
    print(f"\n[3/4] Loading labels...")
    try:
        labels_df = pd.read_csv(Config.LABELS_FILE)
        print(f"      Loaded {len(labels_df)} labels")
        logger.log(f"Loaded {len(labels_df)} labels")
    except Exception as e:
        print(f"      Warning: Could not load labels: {e}")
        labels_df = None

    # Process images
    print(f"\n[4/4] Processing {len(image_files)} images...")
    logger.log(f"Starting rotation of {len(image_files)} images")
    print("-" * 80)

    results = []

    for img_path in tqdm(image_files, desc="Rotating"):
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                raise Exception("Could not read image")

            img_name = img_path.name
            original_size = f"{img.shape[1]}x{img.shape[0]}"

            # Predict angle
            angle = rotator.predict_angle(img)

            # Smart rotate with CORRECTED logic
            rotated, correction_angle = rotator.smart_rotate(img, angle)
            rotated_size = f"{rotated.shape[1]}x{rotated.shape[0]}"

            # Save rotated image
            output_path = Config.ALIGNED_DIR / img_name
            cv2.imwrite(str(output_path), rotated)

            # Record results
            result = {
                'filename': img_name,
                'detected_angle': round(angle, 2),
                'correction_angle': round(correction_angle, 2),
                'original_size': original_size,
                'rotated_size': rotated_size,
                'status': 'success'
            }
            results.append(result)

            logger.stats['total_images'] += 1
            logger.stats['successful'] += 1
            logger.stats['angles'].append(angle)
            logger.stats['correction_angles'].append(correction_angle)

        except Exception as e:
            logger.stats['total_images'] += 1
            logger.stats['errors'] += 1
            logger.log(f"ERROR processing {img_path.name}: {e}")
            results.append({
                'filename': img_path.name,
                'error': str(e),
                'status': 'error'
            })

    # Save results
    print(f"\nSaving results...")
    results_df = pd.DataFrame(results)
    output_csv = Config.OUTPUT_DIR / "rotation_results.csv"
    results_df.to_csv(output_csv, index=False)
    logger.log(f"Results saved to: {output_csv}")

    # Save summary
    summary_file = logger.save_summary()

    # Print summary
    print("\n" + "=" * 80)
    print("ROTATION SUMMARY")
    print("=" * 80)

    total = logger.stats['total_images']
    success = logger.stats['successful']
    errors = logger.stats['errors']

    print(f"\nTotal images:     {total}")
    print(f"Successful:       {success} ({success/total*100:.1f}%)")
    print(f"Errors:            {errors} ({errors/total*100:.1f}%)")

    if 'angle_stats' in logger.stats:
        stats = logger.stats['angle_stats']
        print(f"\nDetected Angle Statistics:")
        print(f"  Mean:     {stats['mean']:.2f} degrees")
        print(f"  Std:      {stats['std']:.2f} degrees")
        print(f"  Min:      {stats['min']:.2f} degrees")
        print(f"  Max:      {stats['max']:.2f} degrees")
        print(f"  Median:   {stats['median']:.2f} degrees")

    if 'correction_angle_stats' in logger.stats:
        stats = logger.stats['correction_angle_stats']
        print(f"\nCorrection Angle Statistics:")
        print(f"  Mean:     {stats['mean']:.2f} degrees")
        print(f"  Std:      {stats['std']:.2f} degrees")
        print(f"  Min:      {stats['min']:.2f} degrees")
        print(f"  Max:      {stats['max']:.2f} degrees")
        print(f"  Median:   {stats['median']:.2f} degrees")

    print("\n" + "=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)
    print(f"Rotated images:  {Config.ALIGNED_DIR}")
    print(f"Results CSV:     {output_csv}")
    print(f"Summary JSON:    {summary_file}")
    print(f"Log file:        {logger.log_file}")

    print("\n" + "=" * 80)
    print("[OK] ROTATION COMPLETED!")
    print("=" * 80)
    print(f"\nRotated images saved to: {Config.ALIGNED_DIR}")
    print(f"Total {success} images rotated successfully!")
    print(f"\nCORRECTED LOGIC: Rotate counter-clockwise by normalized angle")
    print("=" * 80)

if __name__ == "__main__":
    main()
