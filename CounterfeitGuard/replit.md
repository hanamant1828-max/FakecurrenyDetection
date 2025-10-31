# Fake Currency Detector Web App

## Overview
A CNN-based web application that detects whether a currency note is genuine or fake using deep learning. The app uses transfer learning with MobileNet architecture and provides Grad-CAM visualizations to highlight suspicious regions on the currency notes.

## Project Architecture
- **Backend**: Flask API with TensorFlow/Keras for ML inference
- **Model**: MobileNet-based CNN with custom classification layers
- **Visualization**: Grad-CAM heatmap overlays on input images
- **Frontend**: Simple HTML/CSS interface for image uploads

## Recent Changes
- **Grad-CAM Layer Fix** (Oct 31, 2025)
  - Fixed Grad-CAM function to auto-detect the correct convolutional layer
  - Updated make_gradcam_heatmap to access layers within MobileNetV2 base model
  - App now handles both trained and demo models correctly
- **Replit Environment Setup** (Oct 31, 2025)
  - Imported from GitHub and configured for Replit
  - Python 3.11 module installed
  - All dependencies installed (TensorFlow 2.15, Flask 3.0, OpenCV, etc.)
  - Workflow configured to run Flask app on port 5000 with webview
  - Deployment configuration set up (autoscale mode)
  - Created model and uploads directories
  - MobileNetV2 pretrained weights downloaded successfully
  - Demo model created (no trained model yet - needs dataset)
  - Web interface verified and working
  - Root .gitignore added for Python artifacts
- **Previous Development** (Before import)
  - Model architecture created with MobileNetV2 transfer learning
  - Training scripts completed (train_model.py)
  - Grad-CAM visualization implemented
  - Web interface designed

## Tech Stack
- Python 3.11
- TensorFlow 2.15 (CNN model with MobileNet transfer learning)
- Flask 3.0 (REST API)
- OpenCV (image preprocessing)
- NumPy, Pillow, Matplotlib (data processing and visualization)

## Features
1. Binary classification: genuine vs fake currency
2. Transfer learning using MobileNet
3. Data augmentation for improved accuracy
4. Grad-CAM visualization for explainability
5. Confidence score display
6. Simple web interface for image upload

## Target
- Model accuracy: >96%
