"""
Download fake currency dataset from Kaggle and organize training data
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import kagglehub

def download_fake_currency_dataset():
    """Download fake currency dataset from Kaggle"""
    print("="*60)
    print("Downloading Indian Currency Dataset from Kaggle...")
    print("="*60)
    
    try:
        path = kagglehub.dataset_download("jagtaranacademy/indian-currency-dataset")
        print(f"\nDataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def organize_dataset(kaggle_path, genuine_images_path):
    """
    Organize dataset into train/val structure with fake/genuine classes
    
    Args:
        kaggle_path: Path to downloaded Kaggle dataset
        genuine_images_path: Path to genuine currency images
    """
    print("\n" + "="*60)
    print("Organizing Dataset for Training...")
    print("="*60)
    
    # Create dataset structure
    dataset_root = "dataset_training"
    for split in ['train', 'val']:
        for class_name in ['fake', 'genuine']:
            os.makedirs(os.path.join(dataset_root, split, class_name), exist_ok=True)
    
    # Collect genuine images
    genuine_dir = Path(genuine_images_path)
    genuine_images = list(genuine_dir.glob("**/*.jpg")) + list(genuine_dir.glob("**/*.png"))
    print(f"\nFound {len(genuine_images)} genuine images")
    
    # Collect fake images from Kaggle dataset
    kaggle_dir = Path(kaggle_path) if kaggle_path else None
    fake_images = []
    
    if kaggle_dir and kaggle_dir.exists():
        fake_images = list(kaggle_dir.glob("**/*.jpg")) + list(kaggle_dir.glob("**/*.png"))
        print(f"Found {len(fake_images)} potential fake images from Kaggle")
    
    # If no fake images from Kaggle, create synthetic fake images
    if len(fake_images) == 0:
        print("\nWarning: No fake images found. Using image augmentation to create variations...")
        fake_images = create_augmented_fakes(genuine_images[:50])
        print(f"Created {len(fake_images)} augmented variations")
    
    # Split genuine images: 80% train, 20% val
    if len(genuine_images) > 0:
        genuine_train, genuine_val = train_test_split(
            genuine_images, test_size=0.2, random_state=42
        )
        
        print(f"\nOrganizing genuine images:")
        print(f"  - Training: {len(genuine_train)} images")
        print(f"  - Validation: {len(genuine_val)} images")
        
        for img in genuine_train:
            shutil.copy(img, f"{dataset_root}/train/genuine/{img.name}")
        
        for img in genuine_val:
            shutil.copy(img, f"{dataset_root}/val/genuine/{img.name}")
    
    # Split fake images: 80% train, 20% val
    if len(fake_images) > 0:
        fake_train, fake_val = train_test_split(
            fake_images, test_size=0.2, random_state=42
        )
        
        print(f"\nOrganizing fake images:")
        print(f"  - Training: {len(fake_train)} images")
        print(f"  - Validation: {len(fake_val)} images")
        
        for img in fake_train:
            dest_path = f"{dataset_root}/train/fake/{img.name}"
            if Path(img).exists():
                shutil.copy(img, dest_path)
        
        for img in fake_val:
            dest_path = f"{dataset_root}/val/fake/{img.name}"
            if Path(img).exists():
                shutil.copy(img, dest_path)
    
    print("\n" + "="*60)
    print("Dataset Organization Complete!")
    print("="*60)
    print(f"\nDataset location: {dataset_root}/")
    print("\nStructure:")
    for split in ['train', 'val']:
        for class_name in ['fake', 'genuine']:
            path = f"{dataset_root}/{split}/{class_name}"
            count = len(os.listdir(path)) if os.path.exists(path) else 0
            print(f"  {split}/{class_name}: {count} images")
    
    return dataset_root

def create_augmented_fakes(genuine_images):
    """
    Create fake variations using image augmentation
    This is a fallback if no real fake images are available
    """
    import cv2
    import numpy as np
    
    fake_images = []
    fake_dir = Path("dataset_training/temp_fake")
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_path in enumerate(genuine_images):
        if not img_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Apply various distortions to simulate fake currency
        # 1. Color shift
        fake1 = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        fake1_path = fake_dir / f"fake_color_{i}.jpg"
        cv2.imwrite(str(fake1_path), fake1)
        fake_images.append(fake1_path)
        
        # 2. Blur
        fake2 = cv2.GaussianBlur(img, (7, 7), 0)
        fake2_path = fake_dir / f"fake_blur_{i}.jpg"
        cv2.imwrite(str(fake2_path), fake2)
        fake_images.append(fake2_path)
        
        # 3. Noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        fake3 = cv2.add(img, noise)
        fake3_path = fake_dir / f"fake_noise_{i}.jpg"
        cv2.imwrite(str(fake3_path), fake3)
        fake_images.append(fake3_path)
    
    return fake_images

if __name__ == "__main__":
    # Download fake currency dataset from Kaggle
    print("\nStep 1: Downloading fake currency dataset from Kaggle...")
    kaggle_dataset_path = download_fake_currency_dataset()
    
    # Path to genuine images uploaded by user
    genuine_images_path = "dataset/genuine_500/500"
    
    # Organize all images
    print("\nStep 2: Organizing dataset...")
    dataset_path = organize_dataset(kaggle_dataset_path, genuine_images_path)
    
    print(f"\n✅ Dataset ready for training!")
    print(f"📁 Location: {dataset_path}/")
    print(f"\n▶️  Next step: Run training script")
    print(f"   python train_currency_model.py")
