
"""
Train model specifically for 500 Rupee notes detection
"""
import os
import shutil
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def create_500_dataset():
    """Create dataset with only 500 rupee notes"""
    print("="*70)
    print("CREATING 500 RUPEE DATASET")
    print("="*70)
    
    # Source directories
    genuine_source = "Test Images/genuine"
    fake_source = "Test Images/fake"
    
    # Destination
    dataset_root = "dataset_500_only"
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    
    # Create structure
    for split in ['train', 'val']:
        for class_name in ['fake', 'genuine']:
            os.makedirs(os.path.join(dataset_root, split, class_name), exist_ok=True)
    
    # Get 500 rupee images (filter by filename containing "500")
    genuine_images = [f for f in os.listdir(genuine_source) 
                     if f.endswith(('.jpg', '.png', '.jpeg')) and '500' in f]
    # For fake, we'll use all since they're counterfeit training samples
    fake_images = [f for f in os.listdir(fake_source) 
                  if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"\nFound {len(genuine_images)} genuine 500 rupee images")
    print(f"Found {len(fake_images)} fake currency images")
    
    # Split: 80% train, 20% val
    genuine_split = int(len(genuine_images) * 0.8)
    fake_split = int(len(fake_images) * 0.8)
    
    genuine_train = genuine_images[:genuine_split]
    genuine_val = genuine_images[genuine_split:]
    fake_train = fake_images[:fake_split]
    fake_val = fake_images[fake_split:]
    
    print(f"\nTrain: {len(genuine_train)} genuine + {len(fake_train)} fake")
    print(f"Val: {len(genuine_val)} genuine + {len(fake_val)} fake")
    
    # Copy images
    for img in genuine_train:
        shutil.copy2(os.path.join(genuine_source, img), 
                    os.path.join(train_dir, 'genuine', img))
    
    for img in fake_train:
        shutil.copy2(os.path.join(fake_source, img), 
                    os.path.join(train_dir, 'fake', img))
    
    for img in genuine_val:
        shutil.copy2(os.path.join(genuine_source, img), 
                    os.path.join(val_dir, 'genuine', img))
    
    for img in fake_val:
        shutil.copy2(os.path.join(fake_source, img), 
                    os.path.join(val_dir, 'fake', img))
    
    print("\n✓ Dataset created successfully!")
    return dataset_root

def create_model(input_shape=(224, 224, 3), num_classes=2):
    """Create MobileNetV2-based model"""
    print("\nCreating model...")
    
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Model created")
    return model

def fine_tune_model(model, num_layers=20):
    """Fine-tune by unfreezing layers"""
    print(f"\nFine-tuning: unfreezing last {num_layers} layers...")
    
    base_model = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_model = layer
            break
    
    if base_model:
        base_model.trainable = True
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    print("✓ Model ready for fine-tuning")
    return model

def train_500_model():
    """Train model for 500 rupee notes"""
    print("="*70)
    print("TRAINING MODEL FOR 500 RUPEE NOTES")
    print("="*70)
    
    # Create dataset
    dataset_root = create_500_dataset()
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    
    # Create model
    model = create_model()
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nDataset info:")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Classes: {train_generator.class_indices}")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    os.makedirs('CounterfeitGuard/model', exist_ok=True)
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'model/currency_detector_500_best.h5',
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
    print("PHASE 1: Initial Training")
    print("="*70)
    
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tuning
    print("\n" + "="*70)
    print("PHASE 2: Fine-Tuning")
    print("="*70)
    
    model = fine_tune_model(model)
    
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('model/currency_detector.h5')
    model.save('CounterfeitGuard/model/currency_detector.h5')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    # Evaluate
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
    print(f"\nFinal Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    print("\nModel saved to:")
    print("  • model/currency_detector.h5")
    print("  • CounterfeitGuard/model/currency_detector.h5")
    print("="*70)
    
    # Plot history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    if history_fine:
        offset = len(history.history['accuracy'])
        ax1.plot(range(offset, offset + len(history_fine.history['accuracy'])), 
                 history_fine.history['accuracy'], label='Train Acc (Fine-tune)', linewidth=2)
        ax1.plot(range(offset, offset + len(history_fine.history['val_accuracy'])), 
                 history_fine.history['val_accuracy'], label='Val Acc (Fine-tune)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy - 500 Rupee Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    if history_fine:
        offset = len(history.history['loss'])
        ax2.plot(range(offset, offset + len(history_fine.history['loss'])), 
                 history_fine.history['loss'], label='Train Loss (Fine-tune)', linewidth=2)
        ax2.plot(range(offset, offset + len(history_fine.history['val_loss'])), 
                 history_fine.history['val_loss'], label='Val Loss (Fine-tune)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss - 500 Rupee Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model/training_history_500.png', dpi=150)
    print(f"\n✓ Training plot saved to model/training_history_500.png")

if __name__ == '__main__':
    train_500_model()
