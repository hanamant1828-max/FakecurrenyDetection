"""
Complete training by running fine-tuning on the already-trained model
"""
import os
from train_indian_currency_model import fine_tune_model, create_data_generators, plot_training_history
from tensorflow import keras

print("="*70)
print("COMPLETING TRAINING: Fine-Tuning Phase")
print("="*70)

# Load the best model from Phase 1
model_path = 'model/indian_currency_detector_best.h5'
print(f"\nLoading model from {model_path}...")
model = keras.models.load_model(model_path)
print("✓ Model loaded successfully")

# Create data generators
train_dir = 'indian_currency_dataset/train'
val_dir = 'indian_currency_dataset/val'
batch_size = 16

print("\nCreating data generators...")
train_generator, val_generator = create_data_generators(
    train_dir, val_dir, batch_size
)

print(f"\nDataset:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Classes: {train_generator.class_indices}")

# Evaluate initial model
print("\n" + "="*70)
print("Initial Model Performance (Before Fine-Tuning)")
print("="*70)
initial_val_loss, initial_val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy: {initial_val_accuracy*100:.2f}%")
print(f"Validation Loss: {initial_val_loss:.4f}")

# Fine-tune the model
print("\n" + "="*70)
print("PHASE 2: Fine-Tuning (Unfreezing Last Layers)")
print("="*70)

model = fine_tune_model(model, num_layers_to_unfreeze=20)

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

# Fine-tune
history_fine = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save('model/indian_currency_detector.h5')
print("\n✓ Final model saved to model/indian_currency_detector.h5")

# Also save to CounterfeitGuard directory
os.makedirs('CounterfeitGuard/model', exist_ok=True)
model.save('CounterfeitGuard/model/currency_detector.h5')
print("✓ Model copied to CounterfeitGuard/model/currency_detector.h5")

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Validation Loss: {val_loss:.4f}")

print(f"\nImprovement: {(val_accuracy - initial_val_accuracy)*100:+.2f}%")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("Model saved to:")
print("  • model/indian_currency_detector.h5")
print("  • model/indian_currency_detector_best.h5")
print("  • CounterfeitGuard/model/currency_detector.h5")
print("="*70)
