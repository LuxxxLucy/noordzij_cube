#!/usr/bin/env python3
"""
Extract letter 'e' from NoordzijCubeGX.woff2 variable font using fonttools
for a configurable NxNxN grid of weight, contrast, and strokeTransition axes.
Save as 128x128 bitmap images.
"""

from pathlib import Path
import json
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from PIL import Image, ImageDraw, ImageFont
import tempfile
import argparse
from tqdm import tqdm


def convert_woff2_to_ttf(woff2_path):
    """Convert woff2 to TTF format in memory"""
    font = TTFont(str(woff2_path))
    
    # Save to temporary TTF file
    temp_ttf = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
    font.save(temp_ttf.name)
    temp_ttf.close()
    
    return temp_ttf.name


def set_font_variations(font_path, weight=500, contrast=500, stroke=500):
    """Create a font instance with specific variation settings"""
    font = TTFont(font_path)
    
    # Check if font has variation axes
    if 'fvar' not in font:
        print("Not a variable font")
        return None
    
    # Set variations
    variations = {
        'wght': weight,
        'cont': contrast,
        'strk': stroke,
    }
    
    # Create instance
    try:
        instanced_font = instancer.instantiateVariableFont(font, variations)
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
        instanced_font.save(temp_file.name)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"Error creating instance: {e}")
        return None


def render_text_simple(text, font_path, image_size=128):
    """Simple rendering using PIL with converted TTF, focused on the letter"""
    import numpy as np
    
    # Render at larger size first to get accurate bounding box
    render_size = image_size * 4
    temp_img = Image.new('L', (render_size, render_size), color=255)
    draw = ImageDraw.Draw(temp_img)
    
    # Load font at larger size
    font_size = int(render_size * 0.8)
    font = ImageFont.truetype(font_path, font_size)
    
    # Get bounding box and center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (render_size - text_width) // 2 - bbox[0]
    y = (render_size - text_height) // 2 - bbox[1]
    
    # Draw text
    draw.text((x, y), text, font=font, fill=0)
    
    # Convert to numpy array to find actual letter pixels
    img_array = np.array(temp_img)
    
    # Find rows and columns that contain dark pixels (threshold at 250 to catch anti-aliasing)
    dark_pixels = img_array < 250
    rows_with_content = np.any(dark_pixels, axis=1)
    cols_with_content = np.any(dark_pixels, axis=0)
    
    # Find the bounding box of actual letter content
    rows = np.where(rows_with_content)[0]
    cols = np.where(cols_with_content)[0]
    
    if len(rows) > 0 and len(cols) > 0:
        # Get tight bounding box
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]
        
        # Add minimal padding (2% of final image size, scaled to render size)
        padding = int(image_size * 0.02 * 4)
        
        top = max(0, top - padding)
        bottom = min(render_size - 1, bottom + padding)
        left = max(0, left - padding)
        right = min(render_size - 1, right + padding)
        
        # Crop to content
        cropped = temp_img.crop((left, top, right + 1, bottom + 1))
        
        # Make it square by padding the shorter dimension
        width, height = cropped.size
        if width > height:
            new_img = Image.new('L', (width, width), color=255)
            new_img.paste(cropped, (0, (width - height) // 2))
            cropped = new_img
        elif height > width:
            new_img = Image.new('L', (height, height), color=255)
            new_img.paste(cropped, ((height - width) // 2, 0))
            cropped = new_img
        
        # Resize to target size with high-quality resampling
        img = cropped.resize((image_size, image_size), Image.Resampling.LANCZOS)
    else:
        # Fallback if no content found
        img = temp_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    
    return img


def idx_to_param(idx, steps, param_min=0, param_max=1000):
    """
    Convert grid index to parameter value
    For steps=5: 0->0, 1->250, 2->500, 3->750, 4->1000
    For steps=20: 0->0, 1->~52.6, 2->~105.3, ..., 19->1000
    """
    return int(param_min + (idx / (steps - 1)) * (param_max - param_min))


def param_to_normalized(param, param_min=0, param_max=1000):
    """Convert parameter [0, 1000] to normalized [0, 1]"""
    return (param - param_min) / (param_max - param_min)


def main():
    parser = argparse.ArgumentParser(description='Extract Noordzij cube font variations')
    parser.add_argument('--steps', type=int, default=20, 
                        help='Number of steps per axis (default: 20 for 20x20x20 grid)')
    parser.add_argument('--output-dir', type=str, default='dataset/noordzij_cube_full',
                        help='Output directory for images')
    parser.add_argument('--image-size', type=int, default=128,
                        help='Size of output images (default: 128)')
    parser.add_argument('--letter', type=str, default='e',
                        help='Letter to extract (default: e)')
    
    args = parser.parse_args()
    
    # Setup paths
    font_path = Path('./asset/NoordzijCubeGX.woff2')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not font_path.exists():
        font_path = Path('../asset/NoordzijCubeGX.woff2')
        if not font_path.exists():
            print(f"Error: Font file not found")
            return
    
    print(f"Converting {font_path.name} to TTF...")
    
    try:
        # First convert woff2 to ttf
        base_ttf = convert_woff2_to_ttf(font_path)
        print(f"Converted to: {base_ttf}")
    except Exception as e:
        print(f"Error converting font: {e}")
        return
    
    # Generate NxNxN cube
    steps = args.steps
    metadata = []
    temp_files = []
    
    total_variations = steps ** 3
    print(f"\nGenerating {steps}x{steps}x{steps} = {total_variations:,} variations...")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Use tqdm for progress bar
    with tqdm(total=total_variations, desc="Extracting") as pbar:
        for x in range(steps):
            for y in range(steps):
                for z in range(steps):
                    # Map grid indices to parameter values [0, 1000]
                    weight = idx_to_param(x, steps)
                    contrast = idx_to_param(y, steps)
                    stroke = idx_to_param(z, steps)
                    
                    # Also compute normalized coordinates [0, 1]
                    norm_x = param_to_normalized(weight)
                    norm_y = param_to_normalized(contrast)
                    norm_z = param_to_normalized(stroke)
                    
                    try:
                        # Create instance with variations
                        instance_ttf = set_font_variations(base_ttf, weight, contrast, stroke)
                        
                        if instance_ttf:
                            temp_files.append(instance_ttf)
                            # Render using the instanced font
                            img = render_text_simple(args.letter, instance_ttf, 
                                                    image_size=args.image_size)
                        else:
                            # Fallback: use base font
                            img = render_text_simple(args.letter, base_ttf, 
                                                    image_size=args.image_size)
                        
                        # Save with simple numeric naming
                        filename = f"noordzij_{args.letter}_{x:03d}_{y:03d}_{z:03d}.png"
                        output_path = output_dir / filename
                        img.save(output_path)
                        
                        metadata.append({
                            'filename': filename,
                            'x': x, 'y': y, 'z': z,
                            'weight': weight,
                            'contrast': contrast,
                            'stroke': stroke,
                            'norm_x': float(norm_x),
                            'norm_y': float(norm_y),
                            'norm_z': float(norm_z),
                        })
                        
                    except Exception as e:
                        print(f"\nError processing ({x},{y},{z}): {e}")
                        import traceback
                        traceback.print_exc()
                    
                    pbar.update(1)
    
    # Cleanup temp files
    print("\nCleaning up temporary files...")
    import os
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass
    try:
        os.unlink(base_ttf)
    except:
        pass
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Completed! Extracted {len(metadata):,}/{total_variations:,} letter variations")
    print(f"  Images: {output_dir.absolute()}")
    print(f"  Metadata: {metadata_path.absolute()}")


if __name__ == '__main__':
    main()

