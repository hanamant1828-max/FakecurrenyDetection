# Fake Currency Detector Web App

## Overview
A CNN-based web application that detects whether a currency note is genuine or fake using deep learning. The app uses transfer learning with MobileNet architecture and provides Grad-CAM visualizations to highlight suspicious regions on the currency notes.

## Project Architecture
- **Backend**: Flask API with TensorFlow/Keras for ML inference
- **Model**: MobileNet-based CNN with custom classification layers
- **Visualization**: Grad-CAM heatmap overlays on input images
- **Frontend**: Simple HTML/CSS interface for image uploads

## Recent Changes
- **Model Trained Successfully** (Oct 31, 2025)
  - Training Accuracy: 100%
  - Validation Accuracy: 98.33% (EXCEEDS >96% TARGET!)
  - Dataset: 300 images (240 training, 60 validation)
  - Images: 120 genuine + 120 fake (training), 30 genuine + 30 fake (validation)
- Initial project setup (Oct 31, 2025)
- Created project structure with model, static, and templates directories
- Set up requirements.txt with TensorFlow, Flask, OpenCV, and other dependencies
- Fixed critical normalization pipeline for optimal MobileNetV2 performance
- Installed system dependencies (libglvnd, X11 libraries) for OpenCV support

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
