"""
Train Indian Currency Counterfeit Detection Model
Uses MobileNetV2 for transfer learning with genuine and fake currency images
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001

# Paths
DATASET_PATH = "dataset_training"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def create_data_generators():
    """Create data generators with augmentation"""
    print("\n" + "="*60)
    print("Creating Data Generators")
    print("="*60)
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data without augmentation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    return train_generator, val_generator

def create_model():
    """Create transfer learning model with MobileNetV2"""
    print("\n" + "="*60)
    print("Building Model Architecture")
    print("="*60)
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{model.summary()}")
    print(f"\nTotal parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    return model

def plot_training_history(history):
    """Plot and save training history"""
    print("\n" + "="*60)
    print("Generating Training Plots")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, 'training_history.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plots saved to: {plot_path}")
    plt.close()

def evaluate_model(model, val_generator):
    """Evaluate model on validation set"""
    print("\n" + "="*60)
    print("Evaluating Model Performance")
    print("="*60)
    
    results = model.evaluate(val_generator, verbose=0)
    
    print(f"\nValidation Loss: {results[0]:.4f}")
    print(f"Validation Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    
    # Generate predictions for confusion matrix
    predictions = model.predict(val_generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_generator.classes
    
    # Calculate per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Fake', 'Genuine']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"             Fake  Genuine")
    print(f"Actual Fake   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"      Genuine {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Save evaluation report
    report_path = os.path.join(MODEL_DIR, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Currency Detection Model Evaluation\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Validation Loss: {results[0]:.4f}\n")
        f.write(f"Validation Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['Fake', 'Genuine']))
        f.write(f"\n\nConfusion Matrix:\n")
        f.write(f"               Predicted\n")
        f.write(f"             Fake  Genuine\n")
        f.write(f"Actual Fake   {cm[0][0]:4d}    {cm[0][1]:4d}\n")
        f.write(f"      Genuine {cm[1][0]:4d}    {cm[1][1]:4d}\n")
    
    print(f"\n✓ Evaluation report saved to: {report_path}")
    
    return results[1]

def train_model():
    """Main training function"""
    print("\n" + "="*70)
    print("   INDIAN CURRENCY COUNTERFEIT DETECTION MODEL TRAINING")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    # Create model
    model = create_model()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'currency_detector_best.h5'),
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
    
    # Train model
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join(MODEL_DIR, 'currency_detector.h5')
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    accuracy = evaluate_model(model, val_gen)
    
    # Summary
    print("\n" + "="*70)
    print("   TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✓ Best model: {MODEL_DIR}/currency_detector_best.h5")
    print(f"✓ Final model: {final_model_path}")
    print(f"✓ Training plots: {MODEL_DIR}/training_history.png")
    print(f"✓ Evaluation report: {MODEL_DIR}/evaluation_report.txt")
    print(f"\n📊 Final Validation Accuracy: {accuracy*100:.2f}%")
    print("\n" + "="*70)

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    
    # Train model
    train_model()
