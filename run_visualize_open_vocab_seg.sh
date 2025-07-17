#!/bin/bash
# Usage: bash run_visualize_open_vocab_seg.sh <features_path> <labels> <original_image> [output_dir]


# Hardcoded paths and labels as in the user's example call
FEATURES_PATH="/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/lseg_embed_features/features/DSC03423_features.npy"
LABELS="chair table door wall whiteboard wardrobe floor ceiling"
ORIGINAL_IMAGE="/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images/DSC03423.JPG"
OUTPUT_DIR="/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/vis_open_voc"

python /home/neural_fields/Unified-Lift-Gabor/lang-seg/visualize_open_vocab_seg.py \
    --features_path "$FEATURES_PATH" \
    --labels $LABELS \
    --original_image "$ORIGINAL_IMAGE" \
    --output_path "$OUTPUT_DIR"
