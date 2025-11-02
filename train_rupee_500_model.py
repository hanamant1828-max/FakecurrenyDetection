"""
Train Currency Detection Model Specifically for Indian ₹500 Notes
This script trains from scratch with proper preprocessing and augmentation
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # Small batch size due to limited data
EPOCHS_PHASE1 = 30  # Initial training with frozen base
EPOCHS_PHASE2 = 20  # Fine-tuning phase
TRAIN_DIR = 'dataset_500_only/train'
VAL_DIR = 'dataset_500_only/val'
MODEL_SAVE_PATH = 'CounterfeitGuard/model/currency_detector.h5'
CHECKPOINT_PATH = 'CounterfeitGuard/model/rupee_500_checkpoint.h5'

print("="*70)
print("TRAINING INDIAN ₹500 NOTE COUNTERFEIT DETECTION MODEL")
print("="*70)

# Verify dataset structure
print(f"\nDataset directories:")
print(f"  Training: {TRAIN_DIR}")
print(f"  Validation: {VAL_DIR}")

# Count images
train_fake = len(os.listdir(os.path.join(TRAIN_DIR, 'fake')))
train_genuine = len(os.listdir(os.path.join(TRAIN_DIR, 'genuine')))
val_fake = len(os.listdir(os.path.join(VAL_DIR, 'fake')))
val_genuine = len(os.listdir(os.path.join(VAL_DIR, 'genuine')))

print(f"\nDataset statistics:")
print(f"  Training: {train_fake} fake + {train_genuine} genuine = {train_fake + train_genuine} total")
print(f"  Validation: {val_fake} fake + {val_genuine} genuine = {val_fake + val_genuine} total")
print(f"  Class balance: {(train_fake/(train_fake+train_genuine)*100):.1f}% fake, {(train_genuine/(train_fake+train_genuine)*100):.1f}% genuine")

# Create model save directory
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create CNN model using MobileNetV2 transfer learning
    Optimized for ₹500 note detection
    """
    print("\nBuilding model architecture...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model for initial training
    base_model.trainable = False
    print(f"Base model loaded: {len(base_model.layers)} layers (frozen)")
    
    # Build custom classification head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with dropout for regularization
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer - 2 classes (fake, genuine)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='rupee_500_detector')
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print(f"Model created: {model.count_params():,} total parameters")
    return model, base_model


def create_data_generators():
    """
    Create data generators with augmentation
    CRITICAL: Use same preprocessing as prediction
    """
    print("\nSetting up data generators...")
    
    # Training data with aggressive augmentation (small dataset)
    train_datagen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,  # Currency notes shouldn't be upside down
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation data - only preprocessing, no augmentation
    val_datagen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Training generator: {train_generator.samples} samples")
    print(f"Validation generator: {val_generator.samples} samples")
    print(f"Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator


def train_phase_1(model, train_gen, val_gen):
    """
    Phase 1: Train with frozen base model
    """
    print("\n" + "="*70)
    print("PHASE 1: Training with frozen MobileNetV2 base")
    print("="*70)
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    return history


def train_phase_2(model, base_model, train_gen, val_gen):
    """
    Phase 2: Fine-tune top layers of base model
    """
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("="*70)
    
    # Unfreeze top layers
    base_model.trainable = True
    unfroze_from = len(base_model.layers) - 30  # Unfreeze last 30 layers
    
    for layer in base_model.layers[:unfroze_from]:
        layer.trainable = False
    
    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"Unfroze {trainable_layers} layers from base model")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
    
    # Fine-tune
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    return history


def plot_training_history(history1, history2):
    """Plot training metrics"""
    print("\nGenerating training plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Combine histories
    metrics = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }
    
    for key in metrics.keys():
        if key in history1.history:
            metrics[key].extend(history1.history[key])
        if key in history2.history:
            metrics[key].extend(history2.history[key])
    
    epochs = range(1, len(metrics['accuracy']) + 1)
    
    # Accuracy
    axes[0, 0].plot(epochs, metrics['accuracy'], 'b-', label='Training', linewidth=2)
    axes[0, 0].plot(epochs, metrics['val_accuracy'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, metrics['loss'], 'b-', label='Training', linewidth=2)
    axes[0, 1].plot(epochs, metrics['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Phase separation line
    phase1_end = len(history1.history.get('accuracy', []))
    for ax in axes.flat:
        if phase1_end > 0:
            ax.axvline(x=phase1_end, color='green', linestyle='--', linewidth=2, label='Fine-tuning starts', alpha=0.7)
    
    plt.tight_layout()
    plot_path = 'CounterfeitGuard/model/rupee_500_training.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Training plot saved to: {plot_path}")
    plt.close()


def evaluate_model(model, val_gen):
    """Evaluate final model performance"""
    print("\n" + "="*70)
    print("FINAL MODEL EVALUATION")
    print("="*70)
    
    results = model.evaluate(val_gen, verbose=1)
    
    print(f"\nValidation Results:")
    for metric, value in zip(model.metrics_names, results):
        print(f"  {metric}: {value:.4f}")
    
    return results


def main():
    """Main training pipeline"""
    print("\nStarting training pipeline...")
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    # Create model
    model, base_model = create_model()
    
    # Phase 1: Train with frozen base
    history1 = train_phase_1(model, train_gen, val_gen)
    
    # Phase 2: Fine-tune
    history2 = train_phase_2(model, base_model, train_gen, val_gen)
    
    # Plot training history
    plot_training_history(history1, history2)
    
    # Final evaluation
    evaluate_model(model, val_gen)
    
    # Save final model
    print(f"\nSaving final model to: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    file_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"Model saved successfully! Size: {file_size:.2f} MB")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel ready for ₹500 note counterfeit detection")
    print(f"Model location: {MODEL_SAVE_PATH}")
    print("\nNext steps:")
    print("1. Test the model with sample ₹500 notes")
    print("2. Verify predictions are not cached between images")
    print("3. Deploy to production environment")


if __name__ == '__main__':
    main()
