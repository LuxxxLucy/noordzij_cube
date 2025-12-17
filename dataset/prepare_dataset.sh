#!/bin/bash
# Full pipeline to extract and split Noordzij cube dataset

set -e  # Exit on error

echo "=================================================="
echo "Noordzij Cube Dataset Preparation Pipeline"
echo "=================================================="
echo ""

# Configuration
STEPS=20
IMAGE_SIZE=128
LETTER="e"

# Step 1: Extract full dataset
echo "Step 1: Extracting ${STEPS}x${STEPS}x${STEPS} grid..."
python scripts/extract_noordzij_full.py \
    --steps $STEPS \
    --output-dir dataset/noordzij_cube_full \
    --image-size $IMAGE_SIZE \
    --letter $LETTER

echo ""
echo "Step 2: Splitting into train/test sets..."
python scripts/split_dataset.py \
    --input-dir dataset/noordzij_cube_full \
    --output-dir dataset \
    --steps $STEPS \
    --sum-tolerance 0.05

echo ""
echo "=================================================="
echo "✓ Dataset preparation complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. Review the split statistics above"
echo "  2. Run training with: python scripts/train_strategic.py"
echo ""

