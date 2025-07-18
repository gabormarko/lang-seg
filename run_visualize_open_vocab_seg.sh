#!/bin/bash
# Usage: bash run_visualize_open_vocab_seg.sh <features_path> <labels> <original_image> [output_dir]



# Hardcoded paths
FEATURES_PATH="/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/lseg_embed_features/features/DSC03642.JPG.npy"
ORIGINAL_IMAGE="/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images/DSC03642.JPG"
OUTPUT_DIR="/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/vis_open_voc"

# Option to use top 100 labels from file
USE_TOP100=${1:-true}

if [ "$USE_TOP100" = "true" ]; then
    LABELS=$(cat /home/neural_fields/Unified-Lift-Gabor/data/scannetpp/top100.txt | tr '\n' ' ')
else
    LABELS="chair table door wall whiteboard wardrobe floor ceiling"
fi

python /home/neural_fields/Unified-Lift-Gabor/lang-seg/visualize_open_vocab_seg.py \
    --features_path "$FEATURES_PATH" \
    --labels $LABELS \
    --original_image "$ORIGINAL_IMAGE" \
    --output_path "$OUTPUT_DIR"
