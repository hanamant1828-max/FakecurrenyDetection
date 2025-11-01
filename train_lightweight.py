"""
Lightweight training script with reduced memory usage
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import sys
sys.path.append('CounterfeitGuard')
from model import create_model, create_data_generators

def train_lightweight_model():
    """Train with reduced memory usage"""
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    
    print("Creating model...")
    model = create_model()
    
    print("\nCreating data generators with smaller batch size...")
    train_generator, val_generator = create_data_generators(
        train_dir, val_dir, batch_size=8
    )
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'CounterfeitGuard/model/currency_detector.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    print("\n" + "="*50)
    print("Starting lightweight training (10 epochs)...")
    print("="*50 + "\n")
    
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save('CounterfeitGuard/model/currency_detector.h5')
    print("\nModel saved to CounterfeitGuard/model/currency_detector.h5")
    
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"\nFinal Validation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    return model, history

if __name__ == '__main__':
    train_lightweight_model()
