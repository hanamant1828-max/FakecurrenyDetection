"""
Training Script for Indian Currency Counterfeit Detection
Trains model on 50, 200, and 500 Rupee notes (genuine vs fake)
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pathlib import Path

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create MobileNetV2-based model for currency detection
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes (2: genuine/fake)
    
    Returns:
        Compiled model
    """
    print("Creating MobileNetV2-based model...")
    
    # Load pre-trained MobileNetV2 (without top layer)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    
    # Preprocessing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model created successfully")
    return model

def fine_tune_model(model, num_layers_to_unfreeze=20):
    """
    Fine-tune the model by unfreezing some layers
    
    Args:
        model: Trained model
        num_layers_to_unfreeze: Number of layers to unfreeze from the end
    
    Returns:
        Model ready for fine-tuning
    """
    print(f"\nFine-tuning: Unfreezing last {num_layers_to_unfreeze} layers...")
    
    # Find the MobileNetV2 base model by searching for it
    base_model = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_model = layer
            break
    
    if base_model is None:
        print("Warning: Could not find MobileNetV2 base model. Skipping fine-tuning.")
        return model
    
    # Unfreeze the last layers
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model prepared for fine-tuning")
    return model

def create_data_generators(train_dir, val_dir, batch_size=16, target_size=(224, 224)):
    """
    Create data generators with augmentation
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        batch_size: Batch size
        target_size: Target image size
    
    Returns:
        Training and validation generators
    """
    print("Creating data generators with augmentation...")
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data (no augmentation)
    val_datagen = ImageDataGenerator()
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✓ Data generators created")
    print(f"  Classes: {train_generator.class_indices}")
    return train_generator, val_generator

def plot_training_history(history, history_fine=None, save_path='model/training_history.png'):
    """Plot and save training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    
    if history_fine:
        offset = len(history.history['accuracy'])
        ax1.plot(range(offset, offset + len(history_fine.history['accuracy'])), 
                 history_fine.history['accuracy'], label='Train Acc (Fine-tune)', linewidth=2)
        ax1.plot(range(offset, offset + len(history_fine.history['val_accuracy'])), 
                 history_fine.history['val_accuracy'], label='Val Acc (Fine-tune)', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    
    if history_fine:
        offset = len(history.history['loss'])
        ax2.plot(range(offset, offset + len(history_fine.history['loss'])), 
                 history_fine.history['loss'], label='Train Loss (Fine-tune)', linewidth=2)
        ax2.plot(range(offset, offset + len(history_fine.history['val_loss'])), 
                 history_fine.history['val_loss'], label='Val Loss (Fine-tune)', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to {save_path}")

def train_model(train_dir='indian_currency_dataset/train', 
                val_dir='indian_currency_dataset/val',
                epochs=15,
                fine_tune_epochs=10,
                batch_size=16):
    """
    Train the Indian currency detection model
    
    Args:
        train_dir: Training directory
        val_dir: Validation directory
        epochs: Number of initial training epochs
        fine_tune_epochs: Number of fine-tuning epochs
        batch_size: Batch size
    
    Returns:
        Trained model and history
    """
    print("="*70)
    print("TRAINING INDIAN CURRENCY COUNTERFEIT DETECTION MODEL")
    print("="*70)
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    print(f"Initial epochs: {epochs}")
    print(f"Fine-tune epochs: {fine_tune_epochs}")
    print(f"Batch size: {batch_size}")
    print("="*70)
    
    # Create model
    model = create_model()
    print(f"\nModel architecture:")
    model.summary()
    
    # Create data generators
    train_generator, val_generator = create_data_generators(
        train_dir, val_dir, batch_size
    )
    
    print(f"\nDataset:")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Classes: {train_generator.class_indices}")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'model/indian_currency_detector_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Initial training
    print("\n" + "="*70)
    print("PHASE 1: Initial Training (Frozen Base Model)")
    print("="*70)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n" + "="*70)
    print("PHASE 2: Fine-Tuning (Unfreezing Last Layers)")
    print("="*70)
    
    model = fine_tune_model(model)
    
    history_fine = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('model/indian_currency_detector.h5')
    print("\n✓ Final model saved to model/indian_currency_detector.h5")
    
    # Also save to CounterfeitGuard directory
    model.save('CounterfeitGuard/model/currency_detector.h5')
    print("✓ Model copied to CounterfeitGuard/model/currency_detector.h5")
    
    # Evaluate
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Plot training history
    plot_training_history(history, history_fine)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Model saved to:")
    print("  • model/indian_currency_detector.h5")
    print("  • CounterfeitGuard/model/currency_detector.h5")
    print("Training history plot:")
    print("  • model/training_history.png")
    print("="*70)
    
    return model, history, history_fine

if __name__ == '__main__':
    # Check if dataset exists
    train_dir = 'indian_currency_dataset/train'
    val_dir = 'indian_currency_dataset/val'
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("="*70)
        print("ERROR: Dataset not found!")
        print("="*70)
        print(f"Expected directories:")
        print(f"  {train_dir}")
        print(f"  {val_dir}")
        print("\nPlease run first:")
        print("  python create_indian_currency_dataset.py")
        print("="*70)
    else:
        # Train the model
        model, history, history_fine = train_model(
            train_dir=train_dir,
            val_dir=val_dir,
            epochs=15,
            fine_tune_epochs=10,
            batch_size=16
        )
