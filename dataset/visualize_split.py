#!/usr/bin/env python3
"""
Visualize the train/test split strategy for the Noordzij cube dataset.
Shows 3D scatter plot of which points are in train vs test.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def is_in_simplex(norm_x, norm_y, norm_z, max_sum=1.0, tolerance=0.01):
    """Check if point is in simplex: x+y+z <= max_sum and x,y,z >= 0"""
    norm_x =  1 - norm_x
    norm_y =  1 - norm_y
    norm_z =  1 - norm_z
    coord_sum = norm_x + norm_y + norm_z
    return coord_sum <= (max_sum + tolerance) and norm_x >= 0 and norm_y >= 0 and norm_z >= 0


def generate_split_visualization(steps=20, sum_tolerance=0.01, output_file='split_visualization.png'):
    """Generate 3D visualization of train/test split"""
    
    # Generate all points
    train_points = []
    test_points = []
    
    for x in range(steps):
        for y in range(steps):
            for z in range(steps):
                # Convert to normalized coordinates
                norm_x = x / (steps - 1)
                norm_y = y / (steps - 1)
                norm_z = z / (steps - 1)
                
                # Check if in training set: simplex x+y+z≤1
                is_train = not is_in_simplex(norm_x, norm_y, norm_z, max_sum=1.0, tolerance=sum_tolerance)
                
                if is_train:
                    train_points.append([norm_x, norm_y, norm_z])
                else:
                    test_points.append([norm_x, norm_y, norm_z])
    
    train_points = np.array(train_points)
    test_points = np.array(test_points)
    
    print(f"Total points: {steps**3:,}")
    print(f"Training points: {len(train_points):,} ({100*len(train_points)/steps**3:.1f}%)")
    print(f"Test points: {len(test_points):,} ({100*len(test_points)/steps**3:.1f}%)")
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot test points (gray, smaller, transparent)
    if len(test_points) > 0:
        ax.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2],
                  c='blue', s=20, alpha=0.6, label='Test')
    
    # Plot train points (blue, larger)
    if len(train_points) > 0:
        ax.scatter(train_points[:, 0], train_points[:, 1], train_points[:, 2],
                  c='gray', s=5, alpha=0.2, label='Train')
    
    # Highlight origin
    ax.scatter([0], [0], [0], c='red', s=100, marker='*', label='Origin', edgecolors='black', linewidths=1)
    
    # Highlight corners
    corners = np.array([
        [1, 0, 0],  # (1,0,0)
        [0, 1, 0],  # (0,1,0)
        [0, 0, 1],  # (0,0,1)
        [1, 1, 1],  # (1,1,1)
    ])
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2],
              c='orange', s=100, marker='s', alpha=0.9, edgecolors='black', linewidths=1)
    
    # Labels and styling
    ax.set_xlabel('Weight (X)')
    ax.set_ylabel('Contrast (Y)')
    ax.set_zlabel('Stroke (Z)')
    ax.set_title('Train/Test Split')
    ax.view_init(elev=30, azim=45)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize train/test split strategy')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of steps per axis (default: 20)')
    parser.add_argument('--sum-tolerance', type=float, default=0.01,
                        help='Tolerance for simplex x+y+z≤1 (default: 0.01)')
    parser.add_argument('--output', type=str, default='split_visualization.png',
                        help='Output file path')
    
    args = parser.parse_args()
    
    generate_split_visualization(
        steps=args.steps,
        sum_tolerance=args.sum_tolerance,
        output_file=args.output
    )


if __name__ == '__main__':
    main()

