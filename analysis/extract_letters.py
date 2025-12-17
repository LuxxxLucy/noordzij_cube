#!/usr/bin/env python3
"""
Extract letter 'e' from all fonts in dataset/lttrset-otf directory
and save as 128x128 bitmap images.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def extract_letter_from_font(font_path, letter='e', image_size=128):
    """
    Render a letter from a font file and return as PIL Image.
    
    Args:
        font_path: Path to the font file
        letter: Character to render (default 'e')
        image_size: Size of the output image (default 128)
    
    Returns:
        PIL Image object
    """
    # Create blank white image
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Try different font sizes to find the best fit
    font_size = image_size
    font = ImageFont.truetype(str(font_path), font_size)
    
    # Get bounding box of the letter
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Scale font to fit 80% of image size (leaving some margin)
    target_size = int(image_size * 0.8)
    scale = min(target_size / text_width, target_size / text_height)
    font_size = int(font_size * scale)
    font = ImageFont.truetype(str(font_path), font_size)
    
    # Recalculate bounding box with new font size
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the text
    x = (image_size - text_width) // 2 - bbox[0]
    y = (image_size - text_height) // 2 - bbox[1]
    
    # Draw the letter in black
    draw.text((x, y), letter, font=font, fill=0)
    
    return img

def main():
    # Setup paths
    dataset_dir = Path('../dataset/lttrset-otf')
    output_dir = Path('output_letters')
    output_dir.mkdir(exist_ok=True)
    
    # Get all OTF font files
    font_files = sorted(dataset_dir.glob('*.otf'))
    
    print(f"Found {len(font_files)} font files")
    print(f"Output directory: {output_dir}")
    
    # Process each font
    success_count = 0
    for i, font_path in enumerate(font_files, 1):
        try:
            # Extract letter 'e'
            img = extract_letter_from_font(font_path, letter='e', image_size=128)
            
            # Save with font name
            output_path = output_dir / f"{font_path.stem}.png"
            img.save(output_path)
            
            success_count += 1
            if i % 50 == 0:
                print(f"Processed {i}/{len(font_files)} fonts...")
                
        except Exception as e:
            print(f"Error processing {font_path.name}: {e}")
    
    print(f"\nCompleted! Successfully extracted {success_count}/{len(font_files)} letters")
    print(f"Images saved to: {output_dir.absolute()}")

if __name__ == '__main__':
    main()

