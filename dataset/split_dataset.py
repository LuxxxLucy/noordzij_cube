#!/usr/bin/env python3
"""
Split the Noordzij cube dataset into train and test sets based on strategic sampling.

Training set includes:
1. Three axes from origin (but not the endpoints):
   - (0,0,0) to (1,0,0): x varies, y=0, z=0
   - (0,0,0) to (0,1,0): x=0, y varies, z=0
   - (0,0,0) to (0,0,1): x=0, y=0, z varies
2. Simplex region: all points where x+y+z ≤ 1 (with x,y,z ≥ 0)

Test set includes:
- Everything else
- Specifically, the interpolation path from (1,0,0) to (1,1,1) is in test
"""

from pathlib import Path
import json
import shutil
import argparse
from tqdm import tqdm


def is_in_simplex(norm_x, norm_y, norm_z, max_sum=1.0, tolerance=0.01):
    """Check if point is in simplex: x+y+z <= max_sum and x,y,z >= 0"""
    norm_x =  1 - norm_x
    norm_y =  1 - norm_y
    norm_z =  1 - norm_z
    coord_sum = norm_x + norm_y + norm_z
    return coord_sum <= (max_sum + tolerance) and norm_x >= 0 and norm_y >= 0 and norm_z >= 0


def analyze_split(train_indices, test_indices, metadata, steps):
    """Analyze and print statistics about the train/test split"""
    print("\n" + "="*60)
    print("DATASET SPLIT ANALYSIS")
    print("="*60)
    
    print(f"\nTotal points: {len(metadata):,}")
    print(f"Training points: {len(train_indices):,} ({100*len(train_indices)/len(metadata):.1f}%)")
    print(f"Test points: {len(test_indices):,} ({100*len(test_indices)/len(metadata):.1f}%)")
    
    # Training set = simplex region
    print("\nTraining set: All points in simplex (x+y+z ≤ 1)")
    
    # Get metadata
    train_meta = [metadata[i] for i in train_indices]
    test_meta = [metadata[i] for i in test_indices]
    
    # Check corners
    corners = {
        '(0,0,0)': (0, 0, 0),
        '(1,0,0)': (steps-1, 0, 0),
        '(0,1,0)': (0, steps-1, 0),
        '(0,0,1)': (0, 0, steps-1),
        '(1,1,0)': (steps-1, steps-1, 0),
        '(1,0,1)': (steps-1, 0, steps-1),
        '(0,1,1)': (0, steps-1, steps-1),
        '(1,1,1)': (steps-1, steps-1, steps-1),
    }
    
    print("\nKey corner points:")
    for name, (x, y, z) in corners.items():
        in_train = any(m['x'] == x and m['y'] == y and m['z'] == z for m in train_meta)
        in_test = any(m['x'] == x and m['y'] == y and m['z'] == z for m in test_meta)
        status = "TRAIN" if in_train else ("TEST" if in_test else "MISSING")
        print(f"  {name}: {status}")
    
    # Check interpolation path from (1,0,0) to (1,1,1)
    # This path has x=steps-1, and y,z vary from 0 to steps-1
    interp_path_in_test = []
    for m in test_meta:
        if m['x'] == steps - 1 and m['y'] == m['z']:  # Diagonal on the x=max face
            interp_path_in_test.append((m['x'], m['y'], m['z']))
    
    print(f"\nInterpolation path (1,0,0) → (1,1,1) points in test: {len(interp_path_in_test)}")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Split Noordzij dataset into train/test')
    parser.add_argument('--input-dir', type=str, default='dataset/noordzij_cube_full',
                        help='Input directory with full dataset')
    parser.add_argument('--output-dir', type=str, default='dataset',
                        help='Output parent directory for train/test splits')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of steps per axis (must match extraction)')
    parser.add_argument('--sum-tolerance', type=float, default=0.01,
                        help='Tolerance for simplex x+y+z≤1 (default: 0.01)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show split statistics without copying files')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    metadata_path = input_dir / 'metadata.json'
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        return
    
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata):,} entries")
    
    # Determine train/test split
    train_indices = []
    test_indices = []
    
    for idx, entry in enumerate(metadata):
        x, y, z = entry['x'], entry['y'], entry['z']
        norm_x = entry['norm_x']
        norm_y = entry['norm_y']
        norm_z = entry['norm_z']
        
        # Check if in training set: simplex x+y+z≤1
        is_train = not is_in_simplex(norm_x, norm_y, norm_z, max_sum=1.0, tolerance=args.sum_tolerance)
        
        if is_train:
            train_indices.append(idx)
        else:
            test_indices.append(idx)
    
    # Analyze split
    analyze_split(train_indices, test_indices, metadata, args.steps)
    
    if args.dry_run:
        print("Dry run complete. No files copied.")
        return
    
    # Create output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'noordzij_train'
    test_dir = output_dir / 'noordzij_test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying files...")
    print(f"  Train: {train_dir}")
    print(f"  Test: {test_dir}")
    
    # Copy training files
    train_metadata = []
    for idx in tqdm(train_indices, desc="Copying train"):
        entry = metadata[idx]
        src = input_dir / entry['filename']
        dst = train_dir / entry['filename']
        shutil.copy2(src, dst)
        train_metadata.append(entry)
    
    # Copy test files
    test_metadata = []
    for idx in tqdm(test_indices, desc="Copying test"):
        entry = metadata[idx]
        src = input_dir / entry['filename']
        dst = test_dir / entry['filename']
        shutil.copy2(src, dst)
        test_metadata.append(entry)
    
    # Save metadata for each split
    with open(train_dir / 'metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    with open(test_dir / 'metadata.json', 'w') as f:
        json.dump(test_metadata, f, indent=2)
    
    print(f"\n✓ Dataset split complete!")
    print(f"  Training: {len(train_metadata):,} samples in {train_dir}")
    print(f"  Test: {len(test_metadata):,} samples in {test_dir}")


if __name__ == '__main__':
    main()

