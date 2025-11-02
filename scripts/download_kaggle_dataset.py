"""
Download and prepare Kaggle dataset for Indian currency counterfeit detection
"""
import os
import subprocess
import zipfile
import shutil
from pathlib import Path

def setup_kaggle_credentials():
    """Set up Kaggle credentials from environment variables"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = {
        "username": os.environ.get('KAGGLE_USERNAME'),
        "key": os.environ.get('KAGGLE_KEY')
    }
    
    kaggle_json_path = kaggle_dir / 'kaggle.json'
    import json
    with open(kaggle_json_path, 'w') as f:
        json.dump(kaggle_json, f)
    
    # Set proper permissions
    os.chmod(kaggle_json_path, 0o600)
    print(f"Kaggle credentials set up at {kaggle_json_path}")

def download_dataset():
    """Download the fake currency dataset from Kaggle"""
    print("\n" + "="*60)
    print("Downloading Indian Fake Currency Dataset from Kaggle...")
    print("="*60 + "\n")
    
    dataset_name = "lekhansaathvik/fake-currency-dataset"
    download_path = "./kaggle_dataset"
    
    # Create download directory
    os.makedirs(download_path, exist_ok=True)
    
    # Download using Kaggle API
    try:
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", dataset_name,
            "-p", download_path,
            "--unzip"
        ], check=True)
        print(f"\n✓ Dataset downloaded successfully to {download_path}")
        return download_path
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        return None

def organize_dataset(download_path):
    """Organize the dataset into train/val structure"""
    print("\n" + "="*60)
    print("Organizing dataset for training...")
    print("="*60 + "\n")
    
    # Check what files are in the download directory
    print(f"Contents of {download_path}:")
    for root, dirs, files in os.walk(download_path):
        level = root.replace(download_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')
    
    # Look for train and validation directories
    train_dir = Path("dataset/train")
    val_dir = Path("dataset/val")
    
    # Create directories
    for split in ['train', 'val']:
        for category in ['fake', 'genuine']:
            os.makedirs(f"dataset/{split}/{category}", exist_ok=True)
    
    # Find and organize the files
    download_path = Path(download_path)
    
    # Try to find the dataset structure
    possible_paths = [
        download_path / "train",
        download_path / "Train",
        download_path / "training",
        download_path,
    ]
    
    # Look for image files and organize them
    train_fake = list(download_path.rglob("*fake*.png")) + list(download_path.rglob("*fake*.jpg")) + list(download_path.rglob("*Fake*.png"))
    train_genuine = list(download_path.rglob("*genuine*.png")) + list(download_path.rglob("*genuine*.jpg")) + list(download_path.rglob("*real*.png"))
    
    # If we can't find organized structure, try the existing indian_currency_dataset
    if os.path.exists("indian_currency_dataset/train"):
        print("\nFound existing indian_currency_dataset! Using that instead...")
        
        # Copy from indian_currency_dataset to dataset
        if os.path.exists("indian_currency_dataset/train/fake"):
            src = "indian_currency_dataset/train"
            dst = "dataset/train"
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"✓ Copied training data from indian_currency_dataset")
        
        if os.path.exists("indian_currency_dataset/val/fake"):
            src = "indian_currency_dataset/val"
            dst = "dataset/val"
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"✓ Copied validation data from indian_currency_dataset")
        
        return True
    
    # If no structured data found, try to organize the downloaded files
    print(f"\nFound {len(train_fake)} fake images and {len(train_genuine)} genuine images")
    
    if len(train_fake) == 0 and len(train_genuine) == 0:
        print("\nWarning: Could not find fake/genuine images in Kaggle dataset")
        print("Please check the dataset structure manually")
        return False
    
    # Split 80-20 for train/val
    from sklearn.model_selection import train_test_split
    
    if train_fake:
        fake_train, fake_val = train_test_split(train_fake, test_size=0.2, random_state=42)
        
        for img_path in fake_train:
            shutil.copy(img_path, f"dataset/train/fake/{img_path.name}")
        
        for img_path in fake_val:
            shutil.copy(img_path, f"dataset/val/fake/{img_path.name}")
        
        print(f"✓ Organized {len(fake_train)} fake training images")
        print(f"✓ Organized {len(fake_val)} fake validation images")
    
    if train_genuine:
        genuine_train, genuine_val = train_test_split(train_genuine, test_size=0.2, random_state=42)
        
        for img_path in genuine_train:
            shutil.copy(img_path, f"dataset/train/genuine/{img_path.name}")
        
        for img_path in genuine_val:
            shutil.copy(img_path, f"dataset/val/genuine/{img_path.name}")
        
        print(f"✓ Organized {len(genuine_train)} genuine training images")
        print(f"✓ Organized {len(genuine_val)} genuine validation images")
    
    return True

def print_dataset_summary():
    """Print summary of organized dataset"""
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60 + "\n")
    
    for split in ['train', 'val']:
        for category in ['fake', 'genuine']:
            path = f"dataset/{split}/{category}"
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{split.capitalize():12} {category.capitalize():10}: {count:4} images")
            else:
                print(f"{split.capitalize():12} {category.capitalize():10}: Directory not found")
    
    print("\n✓ Dataset is ready for training!")
    print("Run: uv run python train_indian_currency_model.py")

if __name__ == '__main__':
    # Set up Kaggle credentials
    setup_kaggle_credentials()
    
    # Download dataset
    download_path = download_dataset()
    
    if download_path:
        # Organize dataset
        success = organize_dataset(download_path)
        
        if success:
            # Print summary
            print_dataset_summary()
        else:
            print("\nNote: Using existing indian_currency_dataset if available")
            print_dataset_summary()
    else:
        print("\nFailed to download dataset. Checking for existing dataset...")
        organize_dataset("./")
        print_dataset_summary()
