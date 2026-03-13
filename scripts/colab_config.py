# ==========================================
# COLAB CONFIGURATION - M4 FINE-TUNING
# ==========================================
# Update these paths to match your Google Drive structure

# Google Drive Paths
DRIVE_BASE_DIR = "/content/drive/MyDrive/Project"

# Data Paths
DRIVE_DATA_DIR = f"{DRIVE_BASE_DIR}/data/data_4digit2"
DRIVE_LABELS_FILE = f"{DRIVE_BASE_DIR}/data/images_4digit2.csv"

# Model Paths
DRIVE_MODEL_PATH = f"{DRIVE_BASE_DIR}/model/M4_OCR.pth"

# Output Paths
DRIVE_OUTPUT_DIR = f"{DRIVE_BASE_DIR}/model/M4_finetuned"

# ==========================================
# TRAINING CONFIGURATION
# ==========================================

# Training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-5  # Very small for fine-tuning

# Layer 2: Sharpening parameters
SHARPEN_STRENGTH = 1.5  # Higher = more sharpening (try 1.0-2.0)
CLAHE_CLIP_LIMIT = 2.0   # Higher = more contrast (try 1.0-4.0)

# Layer 3: Digit weights (higher = more punishment for errors)
# Format: [weight_0, weight_1, ..., weight_9, weight_blank]
#
# 📊 BẢNG TRỌNG SỐ ĐỀ XUẤT (Dựa trên phân tích lỗi thực tế)
# Chữ số | Trọng số | Lý do
#--------|----------|-------
#   6    |   4.0    | Tỉ lệ lỗi 26% (CAO NHẤT) - Target chính!
#   1    |   3.5    | Số lượng lỗi tuyệt đối nhiều nhất (370 lần)
#   8    |   3.0    | Hay bị nhầm thành 0 và 1
#   9    |   2.5    | Dữ liệu ít, dễ bị mô hình bỏ qua
#   7    |   2.0    | Hay nhầm với 1
#   5    |   2.0    | Hay nhầm thành 8 (mirror)
#   4    |   2.0    | Hay nhầm thành 3
#   2    |   1.5    | Tương đối ổn định nhưng cần cải thiện
#   3    |   1.5    | Tương đối ổn định nhưng cần cải thiện
#   0    |   0.5    | GIẢM TRỌNG SỐ - AI bớt "đoán mò" là 0
#
DIGIT_WEIGHTS = [
    0.5,  # 0: GIẢM TRỌNG SỐ - quá phổ biến, AI hay đoán mò
    3.5,  # 1: Số lượng lỗi tuyệt đối nhiều nhất (370 lần)
    1.5,  # 2: Tương đối ổn định
    1.5,  # 3: Tương đối ổn định
    2.0,  # 4: Hay nhầm thành 3
    2.0,  # 5: Hay nhầm thành 8
    4.0,  # 6: TARGET! Tỉ lệ lỗi 26% (cao nhất) - 8x vs 0
    2.0,  # 7: Hay nhầm thành 1
    3.0,  # 8: Hay bị nhầm thành 0 và 1
    2.5,  # 9: Dữ liệu ít, dễ bị bỏ qua
    1.0,  # 10: Blank (CTC) - giữ nguyên
]

# ==========================================
# VALIDATION
# ==========================================

# Train/Val split
TRAIN_RATIO = 0.8

# Target metrics
TARGET_SIX_TO_ZERO_ERROR_RATE = 0.05  # 5%
TARGET_OVERALL_ACCURACY = 0.85         # 85%

# ==========================================
# TIPS & TRICKS
# ==========================================

"""
If training is too slow:
- Reduce BATCH_SIZE to 16
- Reduce NUM_EPOCHS to 5

If 6→0 error rate is still high:
- Increase DIGIT_WEIGHTS[6] to 3.0 or 4.0
- Increase SHARPEN_STRENGTH to 2.0
- Train for more epochs (15-20)

If model is forgetting other digits:
- Decrease LEARNING_RATE to 5e-6
- Reduce DIGIT_WEIGHTS[6] to 1.5
- Check if other digit accuracies are dropping

If CUDA out of memory:
- Reduce BATCH_SIZE to 16 or 8
- Use smaller model (not recommended)
"""
