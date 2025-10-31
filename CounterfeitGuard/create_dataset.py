"""
Create a synthetic currency image dataset for training
Generates realistic-looking currency note images for demonstration
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import random
from pathlib import Path

def create_currency_image(genuine=True, seed=None):
    """
    Generate a synthetic currency note image
    
    Args:
        genuine: If True, create genuine note, else create fake note
        seed: Random seed for reproducibility
    
    Returns:
        PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Create base image (224x224 for MobileNet)
    width, height = 448, 224
    
    # Base colors
    if genuine:
        # Genuine notes have more consistent, vibrant colors
        base_color = random.choice([
            (34, 139, 34),   # Forest green
            (25, 25, 112),   # Midnight blue
            (139, 69, 19),   # Saddle brown
            (128, 0, 128),   # Purple
        ])
        texture_intensity = 0.05
        watermark_alpha = 180
    else:
        # Fake notes have slightly off colors and poor quality
        base_color = random.choice([
            (50, 150, 50),   # Slightly off green
            (40, 40, 120),   # Slightly off blue
            (150, 80, 30),   # Slightly off brown
            (140, 10, 140),  # Slightly off purple
        ])
        texture_intensity = 0.15
        watermark_alpha = 100
    
    # Create base with noise
    img = Image.new('RGB', (width, height), base_color)
    pixels = np.array(img)
    
    # Add texture noise
    noise = np.random.normal(0, texture_intensity * 255, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Add geometric patterns
    num_lines = 15 if genuine else random.randint(8, 12)
    for i in range(num_lines):
        x = random.randint(0, width)
        color = tuple(np.clip(np.array(base_color) + np.random.randint(-30, 30, 3), 0, 255).tolist())
        draw.line([(x, 0), (x, height)], fill=color, width=1)
    
    # Add circles (security features)
    num_circles = random.randint(3, 6) if genuine else random.randint(1, 3)
    for _ in range(num_circles):
        x, y = random.randint(20, width-20), random.randint(20, height-20)
        r = random.randint(10, 30)
        circle_color = tuple(np.clip(np.array(base_color) + 50, 0, 255).tolist())
        draw.ellipse([x-r, y-r, x+r, y+r], outline=circle_color, width=2)
    
    # Add denomination number
    denomination = random.choice([10, 20, 50, 100])
    try:
        font_size = 60 if genuine else random.randint(50, 65)
        
        # Add number in corners
        for pos in [(30, 30), (width-80, 30), (30, height-90), (width-80, height-90)]:
            text_color = (255, 255, 255, watermark_alpha)
            shadow_offset = 2
            # Shadow
            draw.text((pos[0]+shadow_offset, pos[1]+shadow_offset), str(denomination), 
                     fill=(0, 0, 0, watermark_alpha//2))
            # Main text
            draw.text(pos, str(denomination), fill=text_color)
    except:
        pass
    
    # Add watermark patterns
    if genuine:
        # Genuine notes have clear, consistent watermarks
        for i in range(5):
            x = width // 6 * (i + 1)
            draw.ellipse([x-15, height//2-15, x+15, height//2+15], 
                        outline=(255, 255, 255, 150), width=2)
    else:
        # Fake notes have irregular watermarks
        for i in range(random.randint(2, 4)):
            x = random.randint(50, width-50)
            y = random.randint(50, height-50)
            draw.ellipse([x-10, y-10, x+10, y+10], 
                        outline=(255, 255, 255, 80), width=1)
    
    # Add some blur to fake notes
    if not genuine:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    # Resize to 224x224
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Add final quality variations
    if not genuine:
        # Reduce quality slightly for fake notes
        enhancer = np.random.uniform(0.85, 0.95)
        pixels = np.array(img).astype(float)
        pixels = np.clip(pixels * enhancer, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
    
    return img

def create_dataset(num_images_per_class=150, train_split=0.8):
    """
    Create a complete dataset with train/val splits
    
    Args:
        num_images_per_class: Number of images per class (genuine/fake)
        train_split: Proportion for training set
    """
    print("Creating synthetic currency dataset...")
    print(f"Generating {num_images_per_class} genuine and {num_images_per_class} fake notes")
    
    # Create directory structure
    dataset_dir = Path('dataset')
    for split in ['train', 'val']:
        for category in ['fake', 'genuine']:
            (dataset_dir / split / category).mkdir(parents=True, exist_ok=True)
    
    # Determine split
    num_train = int(num_images_per_class * train_split)
    num_val = num_images_per_class - num_train
    
    # Generate images
    for category, is_genuine in [('genuine', True), ('fake', False)]:
        print(f"\nGenerating {category} notes...")
        
        # Training images
        for i in range(num_train):
            img = create_currency_image(genuine=is_genuine, seed=i)
            img.save(dataset_dir / 'train' / category / f'{category}_{i:04d}.jpg', quality=95)
            if (i + 1) % 50 == 0:
                print(f"  Generated {i+1}/{num_train} training images")
        
        # Validation images
        for i in range(num_val):
            img = create_currency_image(genuine=is_genuine, seed=num_train + i)
            img.save(dataset_dir / 'val' / category / f'{category}_{i:04d}.jpg', quality=95)
        
        print(f"  Completed {category}: {num_train} train, {num_val} val")
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset created successfully!")
    print("="*60)
    print(f"Train - Genuine: {num_train} images")
    print(f"Train - Fake: {num_train} images")
    print(f"Val - Genuine: {num_val} images")
    print(f"Val - Fake: {num_val} images")
    print(f"Total: {num_images_per_class * 2} images")
    print("\nDataset structure:")
    print("  dataset/train/genuine/")
    print("  dataset/train/fake/")
    print("  dataset/val/genuine/")
    print("  dataset/val/fake/")

if __name__ == '__main__':
    # Create dataset with 300 images (150 genuine + 150 fake)
    create_dataset(num_images_per_class=150, train_split=0.8)
    print("\nYou can now run: python train_model.py")
