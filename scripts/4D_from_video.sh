#!/bin/bash

# Script to convert video input to 4D animation
# Usage: ./scripts/4D_from_video.sh <input_video_path> [--split_only]
# ./scripts/4D_from_video.sh ./examples/tiger.mp4
# ./scripts/4D_from_video.sh ./examples/tiger.mp4 --split_only
set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_video_path> [--split_only]"
    echo "Example: $0 ./examples/tiger.mp4"
    echo "        $0 ./examples/tiger.mp4 --split_only"
    exit 1
fi

INPUT_VIDEO="$1"

# Default: split_only is "false"
SPLIT_ONLY="false"
if [ $# -ge 2 ]; then
    # Accept split_only as true if the second arg is "split_only" or "true"
    if [ "$2" = "--split_only" ]; then
        SPLIT_ONLY="true"
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_BASE_DIR="$(dirname "$INPUT_VIDEO")"

# Validate input video
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video not found: $INPUT_VIDEO"
    exit 1
fi

# Set up paths
VIDEO_NAME=$(basename "$INPUT_VIDEO" | sed 's/\.[^.]*$//')
PROCESSED_DIR="$OUTPUT_BASE_DIR/${VIDEO_NAME}_processed"
MASKED_RGB_DIR="$PROCESSED_DIR/masked_rgb"
ANIMATION_OUTPUT_DIR="$PROCESSED_DIR/animation"

echo "=========================================="
echo "Processing video: $INPUT_VIDEO"
echo "Base directory: $OUTPUT_BASE_DIR"
echo "Processed directory: $PROCESSED_DIR"
echo "Split only: $SPLIT_ONLY"
echo "=========================================="

# Step 1: Remove background and extract frames or only split frames
echo ""
if [ "$SPLIT_ONLY" = "true" ]; then
    echo "Step 1: Extracting frames only (split_only mode)..."
    python "$PROJECT_ROOT/utils/rmbg_for_black_bg.py" --input "$INPUT_VIDEO" --split_only
    echo "Split only mode done. Exiting pipeline after video frame extraction."
else
    echo "Step 1: Removing background and extracting frames..."
    python "$PROJECT_ROOT/utils/rmbg_for_black_bg.py" --input "$INPUT_VIDEO"
fi

HUNYUAN_INPUT_DIR="$OUTPUT_BASE_DIR/${VIDEO_NAME}_processed/Hunyuan_Gen_Input"
# Step 2: Generate Hunyuan mesh
echo ""
echo "Step 2: Generating Hunyuan mesh..."
FBX_HUNYUAN_O="$(find "$HUNYUAN_INPUT_DIR" -name "*_hunyuan_original_wo_remap_converted.fbx" | head -1)"
if [ -f "$FBX_HUNYUAN_O" ]; then
    echo "Hunyuan mesh already exists at $FBX_HUNYUAN_O, skipping generation."
else
    export HF_ENDPOINT=https://hf-mirror.com
    python "$PROJECT_ROOT/scripts/hunyuan_Gen.py" \
        --root "$OUTPUT_BASE_DIR" \
        --output "$OUTPUT_BASE_DIR" \
        --N 1 \
        --n 0 \
        --skip 256 \
        --seed 42
fi

python "$PROJECT_ROOT/utils/convert_fbx.py" "$HUNYUAN_INPUT_DIR"
# Step 3: Find the generated FBX file
FBX_FILE=$(find "$HUNYUAN_INPUT_DIR" -name "*_hunyuan_original_wo_remap_converted.fbx" | head -1)
if [ -z "$FBX_FILE" ]; then
    echo "Error: FBX file not found in $HUNYUAN_INPUT_DIR"
    exit 1
fi

# Step 4: Generate animation
echo ""
echo "Step 3: Generating animation..."
python "$PROJECT_ROOT/scripts/inference_with_video_only.py" \
    --config="$PROJECT_ROOT/configs/dyscene.yaml" \
    training.resume_ckpt="$PROJECT_ROOT/experiments/checkpoints/ckpt_0000000000060000.pt" \
    model.class_name=model.Pcd_motion.Motion_Latent_Model \
    data_dir="$FBX_FILE" \
    image_path="$MASKED_RGB_DIR" \
    training.num_shape_samples=16384 \
    training.num_pcd_samples=4096 \
    training.frames=256 \
    start_frame=0 \
    training.use_fbx=True \
    output_dir="$ANIMATION_OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Processing complete!"
echo "Animation output: $ANIMATION_OUTPUT_DIR"
echo "=========================================="

