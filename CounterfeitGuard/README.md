# Fake Currency Detector

AI-powered web application that detects whether a currency note is genuine or fake using deep learning.

## Features

- **CNN-based Detection**: Uses MobileNetV2 transfer learning for binary classification
- **High Accuracy**: Designed to achieve >96% accuracy with proper training data
- **Grad-CAM Visualization**: Highlights suspicious regions on the currency note
- **Confidence Scores**: Shows probability percentages for genuine vs fake
- **Simple Web Interface**: Easy-to-use upload and analyze interface
- **Real-time Analysis**: Quick predictions with visual feedback

## Tech Stack

- **Backend**: Flask 3.0
- **ML Framework**: TensorFlow 2.15 with Keras
- **Model**: MobileNetV2 (transfer learning)
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Matplotlib, Grad-CAM
- **Frontend**: HTML, CSS, JavaScript

## Project Structure

```
.
├── app.py                  # Flask API with /predict endpoint
├── model.py                # CNN model architecture and utilities
├── train_model.py          # Training script for the model
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── model/
│   └── currency_detector.h5  # Trained model (created after training)
└── uploads/               # Temporary storage for uploaded images
```

## Installation

The required packages are already installed in this Replit environment:

- tensorflow==2.15.0
- flask==3.0.0
- opencv-python==4.8.1.78
- numpy==1.24.3
- pillow==10.1.0
- matplotlib==3.8.2
- werkzeug==3.0.1

## Usage

### Running the Application

The Flask app is already configured to run automatically. Simply:

1. Click the "Run" button or access the webview
2. Upload a currency note image (PNG, JPG, or JPEG)
3. Click "Analyze Currency"
4. View results with confidence scores and Grad-CAM visualization

### API Endpoints

#### `GET /`
Returns the main web interface

#### `POST /predict`
Accepts image file and returns prediction results

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "prediction": "Genuine" or "Fake",
  "confidence": 95.34,
  "is_genuine": true,
  "probabilities": {
    "fake": 4.66,
    "genuine": 95.34
  },
  "gradcam_image": "/gradcam/gradcam_filename.png"
}
```

#### `GET /gradcam/<filename>`
Returns the Grad-CAM visualization image

## Training Your Own Model

The current model is a demo with random weights. To train with real data:

1. **Prepare Dataset**: Organize images into this structure:
   ```
   dataset/
   ├── train/
   │   ├── fake/       # Fake currency images
   │   └── genuine/    # Genuine currency images
   └── val/
       ├── fake/       # Validation fake images
       └── genuine/    # Validation genuine images
   ```

2. **Run Training**:
   ```bash
   python train_model.py
   ```

3. **Training Features**:
   - Transfer learning with MobileNetV2
   - Data augmentation (rotation, zoom, brightness, flip)
   - Early stopping to prevent overfitting
   - Learning rate reduction on plateau
   - Model checkpointing (saves best model)
   - Fine-tuning of top layers

4. **Training Outputs**:
   - `model/currency_detector.h5` - Final trained model
   - `model/currency_detector_best.h5` - Best model during training
   - `model/training_history.png` - Training metrics visualization

## Model Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Custom Layers**:
  - Global Average Pooling
  - Dense(256) + ReLU + Dropout(0.5)
  - Dense(128) + ReLU + Dropout(0.3)
  - Dense(2) + Softmax (output)

- **Input Shape**: 224x224x3 RGB images
- **Output**: 2 classes (Fake, Genuine)

## Data Augmentation

The training pipeline includes:
- Rotation (±20 degrees)
- Width/Height shift (20%)
- Shear transformation (20%)
- Zoom (20%)
- Horizontal flip
- Brightness adjustment (80-120%)

## Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) shows:
- Which parts of the currency note the model focused on
- Red/yellow regions indicate high importance
- Blue regions indicate low importance
- Helps identify suspicious features on fake notes

## Important Notes

1. **Demo Model**: The current model has random weights and won't give accurate predictions
2. **Training Data**: You need real currency images (genuine and fake) to train properly
3. **Accuracy Target**: With proper dataset, the model should achieve >96% accuracy
4. **Legal Disclaimer**: This is for educational purposes only. Consult local laws regarding currency image handling

## How It Works

1. **Upload**: User uploads currency note image
2. **Preprocessing**: Image is resized to 224x224 and preprocessed for MobileNet
3. **Prediction**: Model classifies as genuine or fake with confidence score
4. **Grad-CAM**: Generates heatmap showing important regions
5. **Display**: Results shown with overlaid visualization

## Performance Tips

- Use clear, well-lit images
- Ensure the entire note is visible
- Avoid blurry or distorted images
- Higher resolution images work better

## Future Enhancements

- Denomination detection
- Currency type identification
- Batch processing
- REST API documentation
- Model performance metrics dashboard
- Database logging of predictions

## License

Educational project - use responsibly and in accordance with local laws.
