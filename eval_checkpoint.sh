#!/bin/bash
# Usage: bash eval_checkpoint.sh <checkpoint_path>

CHECKPOINT=${1:-checkpoints/your_checkpoint.ckpt}
DATASET=ade20k
DATA_PATH=../datasets
BACKBONE=clip_vitb16_384

python test_lseg.py \
  --weights "$CHECKPOINT" \
  --dataset $DATASET \
  --data-path $DATA_PATH \
  --backbone $BACKBONE

echo "Results saved in outdir_ours/"
