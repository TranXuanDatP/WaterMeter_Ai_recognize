# OCR Training with CRNN + biLSTM + CTC Loss

Training OCR model for 4-digit meter reading recognition using CRNN architecture with biLSTM and CTC Loss.

## 📁 Files Overview

| File | Description | Use Case |
|------|-------------|----------|
| [ocr_crnn_bilstm_training.py](ocr_crnn_bilstm_training.py) | Complete Python training script | Standalone Python execution |
| [ocr_training_colab.ipynb](ocr_training_colab.ipynb) | Jupyter Notebook for Colab | Interactive training in Google Colab |
| [README_OCR_TRAINING.md](README_OCR_TRAINING.md) | This file | Documentation |

## 🎯 Model Architecture

```
Input Image (1, 64, 256) - Grayscale
         ↓
    CNN Backbone
  (5 Convolutional Blocks)
         ↓
  Feature Maps (512, 2, 32)
         ↓
     Reshape & Flatten
         ↓
    biLSTM Layer
  (2 layers, bidirectional)
         ↓
  Fully Connected Layer
         ↓
    Output (T, B, 11)
         ↓
    CTC Loss
```

### Key Components:
- **CNN Feature Extractor**: VGG-like architecture with Batch Normalization
- **biLSTM**: Bidirectional LSTM for sequence modeling
- **CTC Loss**: Handles variable-length sequences without character-level alignment

## 🚀 Quick Start (Google Colab)

### Method 1: Upload ZIP File (Recommended - Fastest)

1. **Open the notebook in Colab**
   - Upload `ocr_training_colab.ipynb` to Google Colab

2. **Upload your ZIP file**
   ```
   Click Files icon (📁) → Upload (📤) → Select: m4_ocr_dataset_black_digits.zip
   ```

3. **Run cells in order**
   - The notebook will **automatically extract** the ZIP file
   - No manual extraction needed!

4. **Start training**
   - All cells are ready to run
   - Just go to Runtime → Run all

### Method 2: Google Drive

1. **Upload ZIP to Google Drive**
   - Upload `m4_ocr_dataset_black_digits.zip` to your Google Drive

2. **Update ZIP path in notebook**
   ```python
   ZIP_FILE = "/content/drive/MyDrive/m4_ocr_dataset_black_digits.zip"
   ```

3. **Mount Drive and run**

## 🔧 Using Python Script

### On Google Colab:

```python
# Upload the script
# Then run:
!python ocr_crnn_bilstm_training.py
```

### On Local Machine:

```bash
# 1. Install dependencies
pip install torch torchvision pandas pillow

# 2. Update paths in Config class:
#    - ZIP_FILE: Path to your zip file
#    - DATA_DIR: Where to extract

# 3. Run
python ocr_crnn_bilstm_training.py
```

## 📊 Dataset Structure

```
m4_ocr_dataset_black_digits/
├── images/
│   ├── crop_meter4_00000_0001e09f7ad5442a832f7b5efb74bf2c.jpg
│   ├── crop_meter4_00001_00027f65538244e89c720c2344fd85f2.jpg
│   └── ...
└── labels.csv
    ├── filename, text
    ├── crop_meter4_00000_0001e09f7ad5442a832f7b5efb74bf2c.jpg, 0187
    └── ...
```

### Important Notes:
- ✅ **Labels are strings**: All labels are explicitly converted to strings to avoid type conflicts
- ✅ **4-digit numbers**: Labels are like "0187", "0246", etc.
- ✅ **Grayscale images**: Images are converted to grayscale for training

## ⚙️ Configuration

### Training Parameters (default):

```python
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
IMG_HEIGHT = 64
IMG_WIDTH = 256
```

### Model Parameters:

```python
NUM_CLASSES = 11  # 0-9 + blank (CTC)
NUM_CHANNELS = 1  # Grayscale
BLANK_IDX = 10    # CTC blank token index
```

## 📈 Training Results

After training, you'll get:

- **best_ocr_model.pth** - Best model based on validation loss
- **checkpoint_epoch_N.pth** - Checkpoints every 10 epochs
- **Training plots** - Loss and accuracy curves
- **Prediction function** - For inference on new images

## 🔮 Making Predictions

### After training:

```python
# Load the model
checkpoint = torch.load('best_ocr_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Predict on new image
result = predict(
    image_path="path/to/image.jpg",
    model=model,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)
print(f"Predicted: {result}")
```

## 📝 Key Features

### ✅ String Label Handling
All labels are properly converted to strings to prevent type conflicts:
```python
# CRITICAL: Ensure text column is string type
df['text'] = df['text'].astype(str)
df['filename'] = df['filename'].astype(str)
```

### ✅ Automatic ZIP Extraction
The script automatically detects and extracts ZIP files:
```python
if Config.AUTO_EXTRACT and os.path.exists(Config.ZIP_FILE):
    extract_zip_data(Config.ZIP_FILE, Config.DATA_DIR)
```

### ✅ Flexible Data Loading
Supports multiple data sources:
- ZIP file upload
- Google Drive
- Pre-extracted data

## 🛠️ Troubleshooting

### Issue: Zip file not found
**Solution**: Upload the ZIP file to the correct path or update `ZIP_FILE` in Config

### Issue: Out of memory
**Solution**: Reduce `BATCH_SIZE` in Config (try 16 or 8)

### Issue: Training is slow
**Solution**:
- Ensure GPU is enabled: `Runtime → Change runtime type → GPU`
- Reduce `NUM_EPOCHS` for testing

### Issue: Labels not loading correctly
**Solution**: The script automatically converts labels to strings. Check that your CSV has `filename,text` columns

## 📦 Dependencies

```bash
pip install torch torchvision pandas pillow
```

For Colab, these are pre-installed.

## 🎓 Next Steps

1. **Hyperparameter Tuning**
   - Try different learning rates
   - Experiment with batch sizes
   - Adjust model architecture

2. **Data Augmentation**
   - Add rotation, scaling, brightness
   - Improve generalization

3. **Model Improvement**
   - Try ResNet backbone
   - Add attention mechanism
   - Experiment with different RNN architectures

4. **Deployment**
   - Export to ONNX
   - Create API endpoint
   - Mobile deployment

## 📄 License

This training script is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- CRNN architecture inspired by ["An End-to-End Trainable Neural OCR Approach..."](https://arxiv.org/abs/1507.05717)
- CTC Loss implementation by PyTorch team

---

**Happy Training! 🚀**

For issues or questions, please check the troubleshooting section above.
