"""
Evaluate the trained Indian Currency Counterfeit Detection Model
"""
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_trained_model(model_path='model/indian_currency_detector_best.h5'):
    """Load the trained model"""
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    return model

def evaluate_on_validation_set(model, val_dir='indian_currency_dataset/val'):
    """Evaluate model on validation set"""
    print("\n" + "="*70)
    print("EVALUATING ON VALIDATION SET")
    print("="*70)
    
    # Create data generator (no augmentation for validation)
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Validation samples: {val_generator.samples}")
    print(f"Classes: {val_generator.class_indices}")
    
    # Evaluate
    print("\nEvaluating...")
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    
    print(f"\n{'='*70}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Loss: {loss:.4f}")
    
    # Get predictions
    print("\nGenerating predictions...")
    predictions = model.predict(val_generator, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = val_generator.classes
    class_labels = list(val_generator.class_indices.keys())
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    print(f"\nConfusion Matrix:")
    print(f"{'='*70}")
    print(f"                 Predicted")
    print(f"               Fake  Genuine")
    print(f"Actual  Fake    {cm[0][0]:3d}     {cm[0][1]:3d}")
    print(f"      Genuine  {cm[1][0]:3d}     {cm[1][1]:3d}")
    
    # Classification report
    print(f"\n{'='*70}")
    print("DETAILED CLASSIFICATION REPORT")
    print(f"{'='*70}")
    print(classification_report(true_classes, predicted_classes, 
                                target_names=class_labels, digits=4))
    
    # Sample predictions
    print(f"\n{'='*70}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*70}")
    filenames = val_generator.filenames
    for i in range(min(12, len(filenames))):
        true_label = class_labels[true_classes[i]]
        pred_label = class_labels[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]] * 100
        status = "✓" if true_label == pred_label else "✗"
        print(f"{status} {filenames[i]:50s} | True: {true_label:8s} | Pred: {pred_label:8s} ({confidence:5.1f}%)")
    
    return accuracy, predictions, true_classes, predicted_classes

def create_confusion_matrix_plot(true_classes, predicted_classes, class_labels):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Indian Currency Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    save_path = 'model/confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {save_path}")

def main():
    """Main evaluation"""
    print("="*70)
    print("INDIAN CURRENCY COUNTERFEIT DETECTION - MODEL EVALUATION")
    print("="*70)
    
    # Load model
    model = load_trained_model()
    
    # Evaluate on validation set
    accuracy, predictions, true_classes, predicted_classes = evaluate_on_validation_set(model)
    
    # Create visualizations
    class_labels = ['fake', 'genuine']
    create_confusion_matrix_plot(true_classes, predicted_classes, class_labels)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"Final Validation Accuracy: {accuracy*100:.2f}%")
    print("\nOutput files:")
    print("  • model/confusion_matrix.png")
    print("="*70)

if __name__ == '__main__':
    main()
