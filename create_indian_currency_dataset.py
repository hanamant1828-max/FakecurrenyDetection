"""
Create Indian Currency Dataset for Counterfeit Detection
Uses genuine currency images and creates fake versions through augmentation
"""
import os
import shutil
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np

def setup_dataset_structure():
    """Create directory structure for Indian currency dataset"""
    print("="*70)
    print("CREATING INDIAN CURRENCY DATASET")
    print("Denominations: 50, 200, 500 Rupees")
    print("="*70)
    
    base_dir = Path('indian_currency_dataset')
    
    # Simple structure: train/genuine, train/fake, val/genuine, val/fake
    for split in ['train', 'val']:
        for category in ['genuine', 'fake']:
            dir_path = base_dir / split / category
            dir_path.mkdir(parents=True, exist_ok=True)
    
    print("\n✓ Directory structure created:")
    print("  indian_currency_dataset/train/genuine/")
    print("  indian_currency_dataset/train/fake/")
    print("  indian_currency_dataset/val/genuine/")
    print("  indian_currency_dataset/val/fake/")
    
    return base_dir

def process_and_resize_image(img_path, target_size=(224, 224)):
    """Resize image to target size for model training"""
    img = Image.open(img_path)
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Resize maintaining aspect ratio
    img.thumbnail((target_size[0], target_size[1]), Image.Resampling.LANCZOS)
    # Create blank canvas and paste centered
    canvas = Image.new('RGB', target_size, (0, 0, 0))
    offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
    canvas.paste(img, offset)
    return canvas

def copy_genuine_images(base_dir):
    """Copy genuine currency images from stock images to dataset"""
    print("\n" + "="*70)
    print("STEP 1: Organizing Genuine Currency Images")
    print("="*70)
    
    stock_dir = Path('attached_assets/stock_images')
    
    if not stock_dir.exists():
        print("⚠️  Stock images directory not found!")
        return 0
    
    # Get all currency images
    currency_images = list(stock_dir.glob('indian_*_rupee*.jpg'))
    
    if not currency_images:
        print("⚠️  No Indian currency images found!")
        return 0
    
    print(f"Found {len(currency_images)} currency images")
    
    # Split into train (80%) and val (20%)
    random.shuffle(currency_images)
    split_idx = int(len(currency_images) * 0.8)
    train_images = currency_images[:split_idx]
    val_images = currency_images[split_idx:]
    
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    
    # Copy and process train images
    train_genuine_dir = base_dir / 'train' / 'genuine'
    for idx, img_path in enumerate(train_images):
        try:
            # Process and resize
            img = process_and_resize_image(img_path)
            # Save
            new_name = f'genuine_{idx:04d}.jpg'
            img.save(train_genuine_dir / new_name, quality=95)
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    # Copy and process val images
    val_genuine_dir = base_dir / 'val' / 'genuine'
    for idx, img_path in enumerate(val_images):
        try:
            img = process_and_resize_image(img_path)
            new_name = f'genuine_{idx:04d}.jpg'
            img.save(val_genuine_dir / new_name, quality=95)
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    total_copied = len(train_images) + len(val_images)
    print(f"\n✓ Processed and copied {total_copied} genuine currency images")
    return total_copied

def create_fake_version(genuine_img_path, augmentation_level='medium'):
    """
    Create a fake currency version using augmentation
    Simulates counterfeit characteristics:
    - Color distortion
    - Blur (poor printing quality)
    - Reduced sharpness
    - Contrast/brightness variations
    """
    img = Image.open(genuine_img_path)
    
    if augmentation_level == 'heavy':
        color_range = (0.6, 1.4)
        blur_range = (0.8, 2.0)
        sharp_range = (0.4, 0.7)
    elif augmentation_level == 'medium':
        color_range = (0.7, 1.3)
        blur_range = (0.5, 1.5)
        sharp_range = (0.5, 0.8)
    else:  # light
        color_range = (0.8, 1.2)
        blur_range = (0.3, 1.0)
        sharp_range = (0.6, 0.9)
    
    # 1. Color shift (counterfeit notes often have off colors)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(*color_range))
    
    # 2. Add blur (lower quality printing)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*blur_range)))
    
    # 3. Reduce sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(random.uniform(*sharp_range))
    
    # 4. Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 5. Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # 6. Add noise
    if random.random() > 0.5:
        pixels = np.array(img).astype(float)
        noise = np.random.normal(0, 5, pixels.shape)
        pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
    
    return img

def generate_fake_images(base_dir, num_augmentations=3):
    """Generate fake currency images from genuine ones using augmentation"""
    print("\n" + "="*70)
    print("STEP 2: Generating Fake Currency Images")
    print("="*70)
    print("Augmentation techniques:")
    print("  • Color distortion (off-color printing)")
    print("  • Gaussian blur (poor print quality)")
    print("  • Sharpness reduction (low-quality materials)")
    print("  • Brightness/contrast variations")
    print("  • Random noise addition")
    print(f"  • Creating {num_augmentations} fake versions per genuine image")
    print("="*70)
    
    total_fake_generated = 0
    
    # Generate fake training images
    train_genuine_dir = base_dir / 'train' / 'genuine'
    train_fake_dir = base_dir / 'train' / 'fake'
    
    genuine_images = list(train_genuine_dir.glob('*.jpg'))
    print(f"\nGenerating fake training images from {len(genuine_images)} genuine images...")
    
    for gen_img in genuine_images:
        for aug_idx in range(num_augmentations):
            try:
                # Create fake version
                fake_img = create_fake_version(gen_img, augmentation_level='medium')
                
                # Save
                base_name = gen_img.stem
                fake_name = f'fake_{base_name}_aug{aug_idx}.jpg'
                fake_img.save(train_fake_dir / fake_name, quality=85)  # Lower quality for fakes
                total_fake_generated += 1
            except Exception as e:
                print(f"  Error creating fake from {gen_img.name}: {e}")
    
    print(f"  ✓ Generated {total_fake_generated} fake training images")
    
    # Generate fake validation images
    val_genuine_dir = base_dir / 'val' / 'genuine'
    val_fake_dir = base_dir / 'val' / 'fake'
    
    genuine_images = list(val_genuine_dir.glob('*.jpg'))
    print(f"\nGenerating fake validation images from {len(genuine_images)} genuine images...")
    
    val_fake_count = 0
    for gen_img in genuine_images:
        for aug_idx in range(num_augmentations):
            try:
                fake_img = create_fake_version(gen_img, augmentation_level='medium')
                base_name = gen_img.stem
                fake_name = f'fake_{base_name}_aug{aug_idx}.jpg'
                fake_img.save(val_fake_dir / fake_name, quality=85)
                val_fake_count += 1
            except Exception as e:
                print(f"  Error creating fake from {gen_img.name}: {e}")
    
    total_fake_generated += val_fake_count
    print(f"  ✓ Generated {val_fake_count} fake validation images")
    
    print(f"\n✓ Total fake images generated: {total_fake_generated}")
    return total_fake_generated

def print_dataset_summary(base_dir):
    """Print summary of created dataset"""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    
    train_genuine = len(list((base_dir / 'train' / 'genuine').glob('*.jpg')))
    train_fake = len(list((base_dir / 'train' / 'fake').glob('*.jpg')))
    val_genuine = len(list((base_dir / 'val' / 'genuine').glob('*.jpg')))
    val_fake = len(list((base_dir / 'val' / 'fake').glob('*.jpg')))
    
    print(f"Training Set:")
    print(f"  Genuine: {train_genuine} images")
    print(f"  Fake:    {train_fake} images")
    print(f"  Total:   {train_genuine + train_fake} images")
    print(f"\nValidation Set:")
    print(f"  Genuine: {val_genuine} images")
    print(f"  Fake:    {val_fake} images")
    print(f"  Total:   {val_genuine + val_fake} images")
    print(f"\nGrand Total: {train_genuine + train_fake + val_genuine + val_fake} images")
    print("="*70)
    
    return {
        'train_genuine': train_genuine,
        'train_fake': train_fake,
        'val_genuine': val_genuine,
        'val_fake': val_fake
    }

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("INDIAN CURRENCY COUNTERFEIT DETECTION DATASET")
    print("Dataset Creator for 50, 200, and 500 Rupee Notes")
    print("="*70)
    
    # Setup directory structure
    base_dir = setup_dataset_structure()
    
    # Copy genuine images
    genuine_count = copy_genuine_images(base_dir)
    
    if genuine_count == 0:
        print("\n⚠️  ERROR: No genuine images found!")
        print("Please ensure stock images are available.")
        return
    
    # Generate fake images (3 variations per genuine image)
    fake_count = generate_fake_images(base_dir, num_augmentations=3)
    
    # Print summary
    summary = print_dataset_summary(base_dir)
    
    # Final instructions
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("Dataset created successfully!")
    print("\nTo train the model, run:")
    print("  python train_indian_currency_model.py")
    print("\nThe trained model will be saved to:")
    print("  model/indian_currency_detector.h5")
    print("="*70)

if __name__ == '__main__':
    main()
