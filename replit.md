# Fake Currency Detector

## Overview

An AI-powered web application that detects whether a currency note is genuine or fake using deep learning. The system uses transfer learning with MobileNetV2 architecture and provides visual explanations through Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps that highlight suspicious regions on currency notes.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Technology**: HTML templates with Jinja2 templating engine
- **Interface Components**: 
  - Main detection page with drag-and-drop file upload
  - Testing dashboard for model debugging and batch testing
  - Authentication pages (login/register)
  - Base template system for consistent navigation and layout
- **Styling**: Custom CSS with modern design patterns (gradients, shadows, responsive cards)
- **Client-side Logic**: JavaScript for file handling, preview generation, and asynchronous API communication

### Backend Architecture
- **Framework**: Flask 3.0 web application
- **Design Pattern**: MVC-style separation with templates, routes, and model logic
- **API Endpoints**:
  - `/predict` - Image classification endpoint
  - Authentication routes (login, register, logout)
  - Testing dashboard route
- **File Handling**: Werkzeug secure filename processing with 16MB upload limit
- **Session Management**: Flask-Login for user authentication with session-based authentication

### Machine Learning Architecture
- **Base Model**: MobileNetV2 pre-trained on ImageNet (transfer learning approach)
- **Model Structure**:
  - Frozen MobileNetV2 base for feature extraction
  - Global average pooling layer
  - Custom classification head with dense layers (256 → 128 neurons)
  - Dropout regularization (0.5 and 0.3) to prevent overfitting
  - Softmax output for binary classification (genuine/fake)
- **Input**: 224x224x3 RGB images
- **Training Strategy**: Two-phase training (frozen base, then fine-tuning)
- **Visualization**: Grad-CAM implementation for explainable AI - highlights image regions influencing predictions

### Data Storage
- **Database**: SQLite with SQLAlchemy ORM
- **Schema**: User model with email and password hash fields
- **Security**: Werkzeug password hashing (generate_password_hash/check_password_hash)
- **File Storage**: Local filesystem for uploaded images and trained models
- **Model Persistence**: Keras H5 format for saved models

### Authentication & Authorization
- **Library**: Flask-Login for session management
- **User Model**: SQLAlchemy-based User class implementing UserMixin
- **Password Security**: Hashed passwords using Werkzeug security utilities
- **Protected Routes**: `@login_required` decorator for restricted endpoints
- **Session**: Server-side session with configurable secret key

## External Dependencies

### Core ML Framework
- **TensorFlow 2.15**: Deep learning framework for model training and inference
- **Keras**: High-level neural networks API (part of TensorFlow)
- **MobileNetV2 Weights**: Pre-trained ImageNet weights downloaded from Keras applications

### Image Processing
- **OpenCV 4.8.1**: Computer vision library for image preprocessing
- **Pillow 10.1.0**: Python Imaging Library for image I/O and manipulation
- **NumPy 1.24.3**: Numerical computing for array operations

### Web Framework
- **Flask 3.0.0**: Lightweight WSGI web application framework
- **Flask-SQLAlchemy**: ORM integration for database operations
- **Flask-Login**: User session management
- **Werkzeug 3.0.1**: WSGI utilities and security helpers

### Visualization
- **Matplotlib 3.8.2**: Plotting library for Grad-CAM heatmap generation

### Training Infrastructure
- **Data Augmentation**: ImageDataGenerator for training data preprocessing and augmentation (rotation, zoom, shift, flip)
- **Synthetic Data Generation**: Custom script for creating demonstration currency images when real datasets unavailable
- **Callbacks**: ModelCheckpoint for saving best models, EarlyStopping for training optimization

### Deployment Environment
- **Platform**: Replit with Python 3.11 runtime
- **Port Configuration**: Flask runs on port 5000 with webview
- **Scaling**: Autoscale deployment mode
- **Environment Variables**: SECRET_KEY for session security