"""
CNN Model for Currency Detection using Transfer Learning with MobileNet
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def create_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create CNN model using MobileNetV2 transfer learning
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes (2 for genuine/fake)
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers for transfer learning
    base_model.trainable = False
    
    # Build custom classification head
    inputs = keras.Input(shape=input_shape)
    
    # No preprocessing here - it's handled by the data generator
    # This ensures training and inference use the same preprocessing
    x = base_model(inputs, training=False)
    
    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout for regularization
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_data_generators(train_dir, val_dir, batch_size=32, img_size=(224, 224)):
    """
    Create data generators with augmentation for training and validation
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        batch_size: Batch size for training
        img_size: Target image size
    
    Returns:
        train_generator, val_generator
    """
    # Training data augmentation
    # Note: No rescaling here - MobileNetV2 preprocessing is applied in the model
    train_datagen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data (only MobileNetV2 preprocessing)
    val_datagen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    return train_generator, val_generator


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess single image for prediction
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    img = keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array


def fine_tune_model(model, base_model_layers=100):
    """
    Fine-tune the model by unfreezing top layers
    
    Args:
        model: Trained model to fine-tune
        base_model_layers: Number of layers to unfreeze from the end
    
    Returns:
        Model ready for fine-tuning
    """
    # Find the MobileNetV2 base model
    base_model = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find MobileNetV2 base model. Skipping fine-tuning.")
        return model
    
    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-base_model_layers]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
