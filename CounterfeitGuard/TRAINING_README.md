# Training Denomination-Specific Currency Detection Models

This guide explains how to train separate AI models for detecting counterfeit Indian currency notes of different denominations (₹100, ₹200, ₹500).

## Dataset Structure

Organize your dataset as follows:

```
dataset_training/
├── 100/
│   ├── train/
│   │   ├── fake/        # Fake ₹100 notes for training
│   │   └── genuine/     # Genuine ₹100 notes for training
│   └── val/
│       ├── fake/        # Fake ₹100 notes for validation
│       └── genuine/     # Genuine ₹100 notes for validation
├── 200/
│   ├── train/
│   │   ├── fake/        # Fake ₹200 notes for training
│   │   └── genuine/     # Genuine ₹200 notes for training
│   └── val/
│       ├── fake/        # Fake ₹200 notes for validation
│       └── genuine/     # Genuine ₹200 notes for validation
└── 500/
    ├── train/
    │   ├── fake/        # Fake ₹500 notes for training
    │   └── genuine/     # Genuine ₹500 notes for training
    └── val/
        ├── fake/        # Fake ₹500 notes for validation
        └── genuine/     # Genuine ₹500 notes for validation
```

## Training Commands

### Train All Denominations

To train models for all denominations at once:

```bash
cd CounterfeitGuard
python train_denomination_models.py --denomination all
```

### Train Individual Denominations

To train a model for a specific denomination:

```bash
# Train ₹500 model only
python train_denomination_models.py --denomination 500

# Train ₹200 model only
python train_denomination_models.py --denomination 200

# Train ₹100 model only
python train_denomination_models.py --denomination 100
```

### Custom Training Parameters

You can customize the training process:

```bash
# Train with 30 epochs and batch size of 16
python train_denomination_models.py --denomination 500 --epochs 30 --batch-size 16

# Train all denominations with custom parameters
python train_denomination_models.py --denomination all --epochs 25 --batch-size 32
```

## Training Output

After training, you'll find:

1. **Trained Models**: Saved in `models/` directory
   - `currency_detector_100.h5` - Model for ₹100 notes
   - `currency_detector_200.h5` - Model for ₹200 notes
   - `currency_detector_500.h5` - Model for ₹500 notes

2. **Training Plots**: Visual representation of training progress
   - `models/training_history_100.png`
   - `models/training_history_200.png`
   - `models/training_history_500.png`

3. **Best Checkpoints**: Best performing models during training
   - `models/currency_detector_100_best.h5`
   - `models/currency_detector_200_best.h5`
   - `models/currency_detector_500_best.h5`

## Model Architecture

Each denomination uses a transfer learning approach with:
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers**: Dense layers with dropout for regularization
- **Output**: Binary classification (Fake vs Genuine)
- **Input Size**: 224×224 pixels

## Data Augmentation

Training uses the following augmentation techniques:
- Random rotation (±20°)
- Width/height shifts (±20%)
- Shear transformations
- Zoom (±20%)
- Horizontal flips
- Brightness adjustments (80%-120%)

## Training Process

1. **Initial Training**: Transfer learning with frozen MobileNetV2 base
2. **Fine-tuning**: Unfreezing top layers for better accuracy
3. **Callbacks**:
   - Early stopping (patience: 5 epochs)
   - Learning rate reduction on plateau
   - Model checkpointing (saves best model)

## Requirements

The application uses the existing dataset in `../dataset_training/` with real currency images organized by denomination.

## Notes

- Training time depends on dataset size and hardware (GPU recommended)
- Minimum recommended: 100+ images per class (fake/genuine) per denomination
- Models are saved automatically and loaded by the web application
- If no trained model exists, the app will use a demo model with random weights
