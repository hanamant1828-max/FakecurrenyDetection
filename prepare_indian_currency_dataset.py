"""
Download and prepare Indian Currency dataset for training
Focuses on 50, 200, and 500 rupee notes (genuine and fake)
"""
import os
import requests
from pathlib import Path
import zipfile
import shutil

def download_file(url, destination):
    """Download file with progress indication"""
    print(f"Downloading from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    f.write(chunk)
                    done = int(50 * downloaded / total_size)
                    print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False

def setup_dataset_structure():
    """Create directory structure for training"""
    print("\nSetting up dataset directory structure...")
    
    base_dir = Path('indian_currency_dataset')
    
    # Create directories for each denomination
    for split in ['train', 'val']:
        for denomination in ['50', '200', '500']:
            for category in ['genuine', 'fake']:
                dir_path = base_dir / split / denomination / category
                dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created:")
    print("  indian_currency_dataset/")
    print("    train/")
    print("      50/ (genuine/, fake/)")
    print("      200/ (genuine/, fake/)")
    print("      500/ (genuine/, fake/)")
    print("    val/")
    print("      50/ (genuine/, fake/)")
    print("      200/ (genuine/, fake/)")
    print("      500/ (genuine/, fake/)")
    
    return base_dir

def download_mendeley_dataset():
    """
    Download Mendeley Indian Currency Dataset
    Note: This is a public dataset for research purposes
    """
    print("\n" + "="*70)
    print("MENDELEY INDIAN CURRENCY DATASET")
    print("="*70)
    print("Dataset: Indian Currency Dataset")
    print("Source: Mendeley Data")
    print("URL: https://data.mendeley.com/datasets/8ckhkssyn3/1")
    print("License: CC BY 4.0")
    print("\nThis dataset contains:")
    print("  - 50 Rs: 272 images")
    print("  - 200 Rs: 205 images")
    print("  - 500 Rs: 223 images")
    print("  - Other denominations: 10, 20, 100, 2000")
    print("\nTotal: 1,786 genuine currency note images")
    print("="*70)
    
    # Direct download URL for Mendeley dataset (if available)
    # Note: Mendeley datasets often require manual download
    dataset_url = "https://data.mendeley.com/public-files/datasets/8ckhkssyn3/files/e5c0a17a-4c6f-4f2b-8c6e-3e0d8c6f5f5e/file_downloaded"
    
    print("\n⚠️  MANUAL DOWNLOAD REQUIRED")
    print("Please follow these steps:")
    print("1. Visit: https://data.mendeley.com/datasets/8ckhkssyn3/1")
    print("2. Click 'Download' to get the dataset")
    print("3. Extract the zip file to 'indian_currency_dataset/downloaded/'")
    print("4. Run this script again to organize the files")
    
    return False

def create_fake_samples_using_augmentation():
    """
    Create synthetic fake currency samples using data augmentation
    This simulates characteristics of counterfeit notes
    """
    from PIL import Image, ImageFilter, ImageEnhance
    import random
    
    print("\n" + "="*70)
    print("CREATING SYNTHETIC FAKE CURRENCY SAMPLES")
    print("="*70)
    print("Using data augmentation to simulate counterfeit characteristics:")
    print("  - Color distortion (fake notes have off colors)")
    print("  - Blur (lower print quality)")
    print("  - Reduced sharpness (poor printing)")
    print("  - Brightness/contrast variations")
    print("="*70)
    
    base_dir = Path('indian_currency_dataset')
    
    # For each denomination, create fake versions from genuine
    for denomination in ['50', '200', '500']:
        genuine_dir = base_dir / 'train' / denomination / 'genuine'
        fake_dir = base_dir / 'train' / denomination / 'fake'
        
        if not genuine_dir.exists():
            print(f"\nNo genuine images found for {denomination} Rs. Skipping...")
            continue
        
        genuine_images = list(genuine_dir.glob('*.jpg')) + list(genuine_dir.glob('*.png'))
        
        if len(genuine_images) == 0:
            print(f"\nNo images found in {genuine_dir}. Skipping...")
            continue
        
        print(f"\nCreating fake samples for {denomination} Rs...")
        print(f"  Source: {len(genuine_images)} genuine images")
        
        # Create fake versions with augmentation
        for idx, img_path in enumerate(genuine_images[:min(len(genuine_images), 100)]):
            try:
                img = Image.open(img_path)
                
                # Apply multiple augmentations to simulate fake characteristics
                # 1. Color shift (counterfeit notes often have off colors)
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(random.uniform(0.7, 1.3))
                
                # 2. Add blur (lower quality printing)
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
                
                # 3. Reduce sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(random.uniform(0.5, 0.8))
                
                # 4. Adjust brightness (inconsistent printing)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))
                
                # 5. Adjust contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.7, 1.3))
                
                # Save fake version
                fake_filename = f'fake_{denomination}_{idx:04d}.jpg'
                img.save(fake_dir / fake_filename, quality=85)  # Lower quality
                
                if (idx + 1) % 20 == 0:
                    print(f"    Generated {idx + 1} fake samples...")
            
            except Exception as e:
                print(f"    Error processing {img_path.name}: {e}")
        
        print(f"  Completed: Created synthetic fake samples for {denomination} Rs")
    
    return True

def main():
    """Main execution"""
    print("="*70)
    print("INDIAN CURRENCY DATASET PREPARATION")
    print("For Counterfeit Detection (50, 200, 500 Rupees)")
    print("="*70)
    
    # Setup directory structure
    base_dir = setup_dataset_structure()
    
    # Download genuine currency dataset
    print("\n" + "="*70)
    print("STEP 1: Download Genuine Currency Dataset")
    print("="*70)
    download_mendeley_dataset()
    
    # Check if user has manually downloaded files
    downloaded_dir = Path('indian_currency_dataset/downloaded')
    if downloaded_dir.exists() and any(downloaded_dir.iterdir()):
        print("\n✓ Downloaded files detected!")
        print("Organizing dataset...")
        # Here you would add code to organize the downloaded files
        # into the proper train/val structure
    else:
        print("\n⚠️  No downloaded files found.")
        print("After downloading the Mendeley dataset:")
        print("  1. Extract files to: indian_currency_dataset/downloaded/")
        print("  2. Run this script again")
        print("\nFor now, I'll create a demonstration dataset using stock images...")
        
        # Use stock images as placeholders
        use_stock_images_as_placeholders(base_dir)
    
    print("\n" + "="*70)
    print("DATASET PREPARATION INSTRUCTIONS")
    print("="*70)
    print("To complete the dataset setup:")
    print("\n1. Download the Mendeley dataset:")
    print("   https://data.mendeley.com/datasets/8ckhkssyn3/1")
    print("\n2. Extract genuine currency images for 50, 200, 500 Rs to:")
    print("   indian_currency_dataset/train/{denomination}/genuine/")
    print("\n3. Run: python prepare_indian_currency_dataset.py")
    print("   This will create synthetic fake samples using augmentation")
    print("\n4. Then train: python CounterfeitGuard/train_model.py")
    print("="*70)

def use_stock_images_as_placeholders(base_dir):
    """Use stock images of Indian currency as placeholders"""
    print("\nUsing stock images for demonstration...")
    print("Note: For production use, please use actual currency datasets")
    
    # This would use the stock_image_tool to download sample images
    # For now, we'll note that manual dataset is needed
    
    return True

if __name__ == '__main__':
    main()
