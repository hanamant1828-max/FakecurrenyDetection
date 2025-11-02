
"""
Create Indian Currency Dataset from Test Images
Uses actual fake and genuine currency images for training
"""
import os
import shutil
from pathlib import Path

def create_dataset():
    """Create train/val split from Test Images directory"""
    
    print("="*70)
    print("CREATING INDIAN CURRENCY DATASET WITH ACTUAL IMAGES")
    print("="*70)
    
    # Source directories
    genuine_source = "Test Images/genuine"
    fake_source = "Test Images/fake"
    
    # Destination directories
    dataset_root = "indian_currency_dataset"
    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    
    # Create directory structure
    for split in ['train', 'val']:
        for class_name in ['fake', 'genuine']:
            os.makedirs(os.path.join(dataset_root, split, class_name), exist_ok=True)
    
    # Get all images
    genuine_images = sorted([f for f in os.listdir(genuine_source) if f.endswith(('.jpg', '.png', '.jpeg'))])
    fake_images = sorted([f for f in os.listdir(fake_source) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"\nFound {len(genuine_images)} genuine currency images")
    print(f"Found {len(fake_images)} fake currency images")
    
    # Split: 80% train, 20% validation
    genuine_split = int(len(genuine_images) * 0.8)
    fake_split = int(len(fake_images) * 0.8)
    
    genuine_train = genuine_images[:genuine_split]
    genuine_val = genuine_images[genuine_split:]
    
    fake_train = fake_images[:fake_split]
    fake_val = fake_images[fake_split:]
    
    print(f"\nTrain set: {len(genuine_train)} genuine + {len(fake_train)} fake")
    print(f"Val set: {len(genuine_val)} genuine + {len(fake_val)} fake")
    
    print("\nCopying images to dataset...")
    
    # Copy training images
    for img in genuine_train:
        src = os.path.join(genuine_source, img)
        dst = os.path.join(train_dir, 'genuine', img)
        shutil.copy2(src, dst)
    
    for img in fake_train:
        src = os.path.join(fake_source, img)
        dst = os.path.join(train_dir, 'fake', img)
        shutil.copy2(src, dst)
    
    # Copy validation images
    for img in genuine_val:
        src = os.path.join(genuine_source, img)
        dst = os.path.join(val_dir, 'genuine', img)
        shutil.copy2(src, dst)
    
    for img in fake_val:
        src = os.path.join(fake_source, img)
        dst = os.path.join(val_dir, 'fake', img)
        shutil.copy2(src, dst)
    
    print("\n" + "="*70)
    print("DATASET CREATED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nDataset structure:")
    print(f"  {train_dir}/fake/ - {len(fake_train)} images")
    print(f"  {train_dir}/genuine/ - {len(genuine_train)} images")
    print(f"  {val_dir}/fake/ - {len(fake_val)} images")
    print(f"  {val_dir}/genuine/ - {len(genuine_val)} images")
    print("="*70)
    
    print("\nNext step: Run 'python train_indian_currency_model.py' to train the model")
    print("="*70)

if __name__ == '__main__':
    create_dataset()
