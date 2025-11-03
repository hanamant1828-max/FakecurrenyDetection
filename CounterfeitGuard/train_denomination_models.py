"""
Training script for Denomination-Specific Currency Detection Models
Train separate models for ₹100, ₹200, and ₹500 notes
"""
import os
import sys
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from model import create_model, create_data_generators, fine_tune_model
import matplotlib.pyplot as plt


DENOMINATIONS = ['100', '200', '500']


def train_denomination_model(denomination, train_dir, val_dir, epochs=20, batch_size=32):
    """
    Train a currency detection model for a specific denomination
    
    Args:
        denomination: Denomination value (e.g., '500', '200', '100')
        train_dir: Directory with training data (subdirectories: fake/, genuine/)
        val_dir: Directory with validation data (subdirectories: fake/, genuine/)
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"Training Model for ₹{denomination} Notes")
    print(f"{'='*60}\n")
    
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    print("Creating model...")
    model = create_model()
    print(f"\nModel architecture summary for ₹{denomination}:")
    model.summary()
    
    print("\nCreating data generators...")
    train_generator, val_generator = create_data_generators(
        train_dir, val_dir, batch_size
    )
    
    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {train_generator.class_indices}")
    
    # Callbacks
    checkpoint_path = f'models/currency_detector_{denomination}_best.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
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
    print(f"Starting training for ₹{denomination}...")
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
    print(f"Fine-tuning model for ₹{denomination}...")
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
    final_model_path = f'models/currency_detector_{denomination}.h5'
    model.save(final_model_path)
    print(f"\nModel saved to {final_model_path}")
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print(f"Evaluating model for ₹{denomination}...")
    print("="*50 + "\n")
    
    val_loss, val_accuracy = model.evaluate(val_generator)
    print(f"\nValidation Accuracy for ₹{denomination}: {val_accuracy*100:.2f}%")
    print(f"Validation Loss for ₹{denomination}: {val_loss:.4f}")
    
    # Plot training history
    plot_training_history(denomination, history, history_fine)
    
    return model, history


def plot_training_history(denomination, history, history_fine=None):
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
    ax1.set_title(f'Model Accuracy - ₹{denomination}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
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
    ax2.set_title(f'Model Loss - ₹{denomination}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'models/training_history_{denomination}.png', dpi=150)
    print(f"Training history plot saved to models/training_history_{denomination}.png")
    plt.close()


def train_all_denominations():
    """
    Train models for all denominations
    
    This function expects the dataset to be organized as:
    dataset_training/
        ├── 100/
        │   ├── train/
        │   │   ├── fake/
        │   │   └── genuine/
        │   └── val/
        │       ├── fake/
        │       └── genuine/
        ├── 200/
        │   ├── train/
        │   │   ├── fake/
        │   │   └── genuine/
        │   └── val/
        │       ├── fake/
        │       └── genuine/
        └── 500/
            ├── train/
            │   ├── fake/
            │   └── genuine/
            └── val/
                ├── fake/
                └── genuine/
    """
    base_dataset_dir = Path('../dataset_training')
    
    if not base_dataset_dir.exists():
        print(f"Error: Dataset directory not found at {base_dataset_dir}")
        print("\nExpected structure:")
        print("dataset_training/")
        print("  ├── 100/train/{fake, genuine}")
        print("  ├── 100/val/{fake, genuine}")
        print("  ├── 200/train/{fake, genuine}")
        print("  ├── 200/val/{fake, genuine}")
        print("  ├── 500/train/{fake, genuine}")
        print("  └── 500/val/{fake, genuine}")
        return
    
    results = {}
    
    for denomination in DENOMINATIONS:
        train_dir = base_dataset_dir / denomination / 'train'
        val_dir = base_dataset_dir / denomination / 'val'
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"\n⚠️  Skipping ₹{denomination}: Dataset not found")
            print(f"   Expected: {train_dir} and {val_dir}")
            continue
        
        try:
            model, history = train_denomination_model(
                denomination=denomination,
                train_dir=str(train_dir),
                val_dir=str(val_dir),
                epochs=20,
                batch_size=32
            )
            results[denomination] = 'Success'
        except Exception as e:
            print(f"\n❌ Error training model for ₹{denomination}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[denomination] = f'Failed: {str(e)}'
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for denom, result in results.items():
        print(f"₹{denom}: {result}")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train denomination-specific currency detection models')
    parser.add_argument('--denomination', type=str, choices=DENOMINATIONS + ['all'], 
                        default='all', help='Denomination to train (or "all" for all denominations)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    if args.denomination == 'all':
        train_all_denominations()
    else:
        base_dataset_dir = Path('../dataset_training')
        train_dir = base_dataset_dir / args.denomination / 'train'
        val_dir = base_dataset_dir / args.denomination / 'val'
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"Error: Dataset not found for ₹{args.denomination}")
            print(f"Expected: {train_dir} and {val_dir}")
            sys.exit(1)
        
        train_denomination_model(
            denomination=args.denomination,
            train_dir=str(train_dir),
            val_dir=str(val_dir),
            epochs=args.epochs,
            batch_size=args.batch_size
        )
