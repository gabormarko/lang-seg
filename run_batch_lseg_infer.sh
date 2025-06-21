#!/bin/bash
# Universal batch script to run batch_lseg_infer.py for a folder of images (no dataset loader)

set -e

cd "$(dirname "$0")"

# --- EDIT THESE VARIABLES FOR YOUR DATASET ---
INPUT_DIR="/home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/images"
OUTPUT_DIR="output/scannetpp_officescene_results"

# ADE20K labels
# LABELS="wall building sky floor tree ceiling road bed windowpane grass
#    cabinet sidewalk person earth door table mountain plant curtain chair
#    car water painting sofa shelf house sea mirror rug field armchair
#    seat fence desk rock wardrobe lamp bathtub railing cushion base
#    box column signboard chest counter sand sink skyscraper fireplace refrigerator
#    grandstand path stairs runway case pool table pillow screen door stairway river
#    bridge bookcase blind coffee table toilet flower book hill bench countertop
#    stove palm kitchen-island computer swivel chair boat bar arcade machine hovel bus
#    towel light truck tower chandelier awning streetlight booth television airplane
#    dirt track apparel pole land bannister escalator ottoman bottle buffet poster
#    stage van ship fountain conveyer belt canopy washer plaything swimming-pool stool
#    barrel basket waterfall tent bag minibike cradle oven ball food step
#    tank tradename microwave pot animal bicycle lake dishwasher screen blanket
#    sculpture hood sconce vase traffic-light tray ashcan fan pier crt screen
#    plate monitor bulletin-board shower radiator glass clock flag"


# ScanNetpp labels (from instance_classes.txt)
LABELS=$(paste -sd' ' /home/neural_fields/Unified-Lift-Gabor/data/scannetpp/officescene/top100.txt)

# --------------------------------------------

if [ -z "$LABELS" ]; then
  python3 batch_lseg_infer.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"
else
  python3 batch_lseg_infer.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR" --labels $LABELS
fi

echo "Batch LSeg inference completed. Results are in $OUTPUT_DIR."
