"""
Training script for Currency Detection Model
This is a sample script - you'll need to provide your own dataset
"""
import os
import tensorflow as tf
from tensorflow import keras
from model import create_model, create_data_generators, fine_tune_model
import matplotlib.pyplot as plt


def train_model(train_dir, val_dir, epochs=20, batch_size=32):
    """
    Train the currency detection model
    
    Args:
        train_dir: Directory with training data (subdirectories: fake/, genuine/)
        val_dir: Directory with validation data (subdirectories: fake/, genuine/)
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained model and training history
    """
    print("Creating model...")
    model = create_model()
    model.summary()
    
    print("\nCreating data generators...")
    train_generator, val_generator = create_data_generators(
        train_dir, val_dir, batch_size
    )
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'model/currency_detector_best.h5',
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
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tune the model
    print("\n" + "="*50)
    print("Fine-tuning model...")
    print("="*50 + "\n")
    
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
    print("\nModel saved to model/currency_detector.h5")
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("Evaluating model...")
    print("="*50 + "\n")
    
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"\nValidation Accuracy: {val_accuracy*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Plot training history
    plot_training_history(history, history_fine)
    
    return model, history


def plot_training_history(history, history_fine=None):
    """Plot training and validation accuracy/loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    
    if history_fine:
        offset = len(history.history['accuracy'])
        ax1.plot(range(offset, offset + len(history_fine.history['accuracy'])), 
                 history_fine.history['accuracy'], label='Train Accuracy (Fine-tune)')
        ax1.plot(range(offset, offset + len(history_fine.history['val_accuracy'])), 
                 history_fine.history['val_accuracy'], label='Val Accuracy (Fine-tune)')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    
    if history_fine:
        offset = len(history.history['loss'])
        ax2.plot(range(offset, offset + len(history_fine.history['loss'])), 
                 history_fine.history['loss'], label='Train Loss (Fine-tune)')
        ax2.plot(range(offset, offset + len(history_fine.history['val_loss'])), 
                 history_fine.history['val_loss'], label='Val Loss (Fine-tune)')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=150)
    print("\nTraining history plot saved to model/training_history.png")


def create_demo_model():
    """
    Create and save a demo model with random weights
    This is just for demonstration - train with real data for actual use
    """
    print("Creating demo model...")
    model = create_model()
    
    # Save demo model
    os.makedirs('model', exist_ok=True)
    model.save('model/currency_detector.h5')
    print("Demo model saved to model/currency_detector.h5")
    print("\nWARNING: This is an untrained model with random weights!")
    print("For real use, train the model with actual currency dataset.")


if __name__ == '__main__':
    # Check if dataset directories exist
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        # Train with actual dataset
        print("Dataset found. Starting training...")
        print(f"Train directory: {train_dir}")
        print(f"Validation directory: {val_dir}")
        print("\nExpected structure:")
        print("  dataset/train/fake/")
        print("  dataset/train/genuine/")
        print("  dataset/val/fake/")
        print("  dataset/val/genuine/\n")
        
        model, history = train_model(train_dir, val_dir, epochs=20)
    else:
        # Create demo model
        print("No dataset found.")
        print(f"Expected directories: {train_dir}, {val_dir}")
        print("\nCreating demo model for testing purposes...")
        create_demo_model()
