"""
Train ₹500 Currency Detection Model with Heavy Data Augmentation
This script trains a CNN model to detect real vs fake Indian ₹500 notes
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATASET_DIR = 'dataset_500_only'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'val')
MODEL_SAVE_PATH = 'CounterfeitGuard/model/currency_detector.h5'
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 30

os.makedirs('CounterfeitGuard/model', exist_ok=True)

print("=" * 70)
print("INDIAN ₹500 CURRENCY DETECTOR - MODEL TRAINING")
print("=" * 70)

train_fake_count = len(os.listdir(os.path.join(TRAIN_DIR, 'fake')))
train_genuine_count = len(os.listdir(os.path.join(TRAIN_DIR, 'genuine')))
val_fake_count = len(os.listdir(os.path.join(VAL_DIR, 'fake')))
val_genuine_count = len(os.listdir(os.path.join(VAL_DIR, 'genuine')))

print(f"\nDataset Statistics:")
print(f"  Training   - Fake: {train_fake_count}, Genuine: {train_genuine_count}, Total: {train_fake_count + train_genuine_count}")
print(f"  Validation - Fake: {val_fake_count}, Genuine: {val_genuine_count}, Total: {val_fake_count + val_genuine_count}")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Max Epochs: {EPOCHS}")

print("\n" + "=" * 70)
print("STEP 1: Creating Data Generators with Heavy Augmentation")
print("=" * 70)

train_datagen = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.6, 1.4],
    channel_shift_range=30.0,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=keras.applications.mobilenet_v2.preprocess_input
)

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
    shuffle=False,
    seed=42
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Training steps per epoch: {len(train_generator)}")
print(f"Validation steps per epoch: {len(val_generator)}")

print("\n" + "=" * 70)
print("STEP 2: Building CNN Model with Transfer Learning")
print("=" * 70)

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False
print(f"Base model (MobileNetV2) loaded with {len(base_model.layers)} frozen layers")

inputs = keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
x = layers.Dense(256, activation='relu', name='dense_256')(x)
x = layers.Dropout(0.5, name='dropout_1')(x)
x = layers.Dense(128, activation='relu', name='dense_128')(x)
x = layers.Dropout(0.3, name='dropout_2')(x)
outputs = layers.Dense(2, activation='softmax', name='output')(x)

model = keras.Model(inputs, outputs, name='rupee_500_detector')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print(f"\nModel Architecture:")
model.summary()

print("\n" + "=" * 70)
print("STEP 3: Setting Up Training Callbacks")
print("=" * 70)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]
print("Callbacks configured: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau")

print("\n" + "=" * 70)
print("STEP 4: Training Model (Phase 1 - Transfer Learning)")
print("=" * 70)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 70)
print("STEP 5: Fine-Tuning (Unfreezing Top Layers)")
print("=" * 70)

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

print(f"Fine-tuning: Unfroze top 30 layers of base model")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

fine_tune_epochs = 20
print(f"Fine-tuning for {fine_tune_epochs} additional epochs...")

history_fine = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 70)
print("STEP 6: Evaluating Final Model")
print("=" * 70)

val_loss, val_accuracy, val_precision, val_recall = model.evaluate(val_generator, verbose=0)
print(f"\nFinal Validation Metrics:")
print(f"  Loss:      {val_loss:.4f}")
print(f"  Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  F1-Score:  {2 * (val_precision * val_recall) / (val_precision + val_recall):.4f}")

print("\n" + "=" * 70)
print("STEP 7: Saving Training History Plot")
print("=" * 70)

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss_history = history.history['val_loss'] + history_fine.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--', label='Fine-tuning starts')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss_history, label='Validation Loss')
plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--', label='Fine-tuning starts')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.tight_layout()
plot_path = 'CounterfeitGuard/model/training_history.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Training history plot saved to: {plot_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nTrained model saved to: {MODEL_SAVE_PATH}")
print(f"Training plot saved to: {plot_path}")
print(f"\nThe model is now ready for deployment!")
print(f"Class mapping: 0=fake, 1=genuine")
print("=" * 70)
