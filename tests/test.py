import os
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

# Thêm đường dẫn src để import utility function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from m2_orientation_alignment.model import sin_cos_to_angle

# ============================================================
# MODEL ARCHITECTURE - Match với saved weights (dùng "head" thay vì "regressor")
# ============================================================

class M2AngleRegressor(nn.Module):
    """M2 Angle Regression CNN - dùng 'head' để khớp với saved weights"""

    def __init__(self, backbone='resnet18', pretrained=False, dropout=0.2):
        super(M2AngleRegressor, self).__init__()

        # Backbone - ResNet18
        weights = 'DEFAULT' if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove classification head

        # Regression head (TEN: head KHONG regressor để khớp với checkpoint)
        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # Output: [sin, cos]
        )

    def forward(self, x):
        features = self.backbone(x)
        sin_cos = self.head(features)
        # Normalize to unit circle
        sin_cos = F.normalize(sin_cos, p=2, dim=1)
        return sin_cos

# ============================================================
# PHẦN 2: HÀM TIỆN ÍCH XỬ LÝ ẢNH
# ============================================================

def smart_rotate(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# ============================================================
# PHẦN 3: CẤU HÌNH ĐƯỜNG DẪN (Dùng đường dẫn tuyệt đối của bạn)
# ============================================================

MODEL_PATH = r"F:\Workspace\Project\model\orientation.pth"
IMAGE_PATH = r"F:\Workspace\Project\data\m2_crops_test\crop_meter4_00069_0305f7a376ce4fcfbb2064368d1e6d96.jpg"

# ============================================================
# PHẦN 4: THỰC THI KIỂM TRA
# ============================================================

def run_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = M2AngleRegressor(backbone='resnet18', pretrained=False).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ LỖI: Không tìm thấy file weights tại {MODEL_PATH}")
        return

    # Load weights
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Xử lý nếu checkpoint là Dictionary chứa 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load vào model
    try:
        model.load_state_dict(state_dict)
        print("✅ Đã load weights thành công!")
    except RuntimeError as e:
        print(f"❌ Lỗi load weights: {e}")
        return

    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ LỖI: Không tìm thấy ảnh tại {IMAGE_PATH}")
        return

    img_bgr = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)

    angle_deg = sin_cos_to_angle(output).cpu().item()
    print(f"AI du doan anh lech: {angle_deg:.2f} do")

    # Xoay nắn thẳng
    aligned_img = smart_rotate(img_bgr, -angle_deg)
    aligned_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB)

    # Hien thi
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"GOC\nLech: {angle_deg:.2f} do")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(aligned_rgb)
    plt.title(f"NAN THANG\nBu: {-angle_deg:.2f} do")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_test()