"""
Create Indian Currency Dataset with ACTUAL fake currency images
Uses real fake currency images instead of augmented genuine images
"""
import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset():
    """Create dataset with actual fake and genuine currency images"""

    # Source directories
    genuine_source = Path('Test Images/genuine')
    fake_source = Path('Test Images/fake')

    # Destination directories
    dataset_dir = Path('indian_currency_dataset')
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'

    # Clean existing dataset
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    # Create directory structure
    for split in ['train', 'val']:
        for cls in ['fake', 'genuine']:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CREATING INDIAN CURRENCY DATASET WITH ACTUAL FAKE IMAGES")
    print("="*70)

    # Process genuine images
    genuine_images = list(genuine_source.glob('*.jpg')) + list(genuine_source.glob('*.png'))
    print(f"\nFound {len(genuine_images)} genuine currency images")

    # Process fake images
    fake_images = list(fake_source.glob('*.jpg')) + list(fake_source.glob('*.png'))
    print(f"Found {len(fake_images)} fake currency images")

    # Split data (80% train, 20% val)
    genuine_train, genuine_val = train_test_split(genuine_images, test_size=0.2, random_state=42)
    fake_train, fake_val = train_test_split(fake_images, test_size=0.2, random_state=42)

    print(f"\nTrain set: {len(genuine_train)} genuine + {len(fake_train)} fake")
    print(f"Val set: {len(genuine_val)} genuine + {len(fake_val)} fake")

    # Copy images to dataset
    print("\nCopying images to dataset...")

    # Copy genuine training images
    for idx, img_path in enumerate(genuine_train):
        dest = train_dir / 'genuine' / f'genuine_{idx:04d}{img_path.suffix}'
        shutil.copy2(img_path, dest)

    # Copy genuine validation images
    for idx, img_path in enumerate(genuine_val):
        dest = val_dir / 'genuine' / f'genuine_{idx:04d}{img_path.suffix}'
        shutil.copy2(img_path, dest)

    # Copy fake training images
    for idx, img_path in enumerate(fake_train):
        dest = train_dir / 'fake' / f'fake_{idx:04d}{img_path.suffix}'
        shutil.copy2(img_path, dest)

    # Copy fake validation images
    for idx, img_path in enumerate(fake_val):
        dest = val_dir / 'fake' / f'fake_{idx:04d}{img_path.suffix}'
        shutil.copy2(img_path, dest)

    print("\n" + "="*70)
    print("DATASET CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\nDataset structure:")
    print(f"  {train_dir}/fake/ - {len(list((train_dir / 'fake').glob('*')))} images")
    print(f"  {train_dir}/genuine/ - {len(list((train_dir / 'genuine').glob('*')))} images")
    print(f"  {val_dir}/fake/ - {len(list((val_dir / 'fake').glob('*')))} images")
    print(f"  {val_dir}/genuine/ - {len(list((val_dir / 'genuine').glob('*')))} images")
    print("="*70)
    print("\nNext step: Run 'python train_indian_currency_model.py' to train the model")
    print("="*70)

if __name__ == '__main__':
    create_dataset()