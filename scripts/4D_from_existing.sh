#!/bin/bash

# Script to generate 4D animation from existing mesh and video
# Usage: ./scripts/4D_from_existing.sh <data_dir> <video_path> <output_dir>
# Example: ./scripts/4D_from_existing.sh ./examples/mesh.glb ./examples/video.mp4 ./examples/output
set -e

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_dir> <video_path> <output_dir>"
    echo "  data_dir: Path to .glb or .fbx mesh file (FBX will be automatically converted to GLB)"
    echo "  video_path: Path to video file (.mp4/.avi/.mov) or directory containing images"
    echo "  output_dir: Output directory for animation results"
    echo ""
    echo "Example: $0 ./examples/chili.glb ./examples/chili.mp4 ./examples/output"
    exit 1
fi

DATA_DIR="$1"
VIDEO_PATH="$2"
OUTPUT_DIR="$3"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validate data_dir
if [ ! -f "$DATA_DIR" ]; then
    echo "Error: Mesh file not found: $DATA_DIR"
    echo "Supported formats: .glb, .fbx"
    exit 1
fi

# Validate video_path
if [ ! -f "$VIDEO_PATH" ] && [ ! -d "$VIDEO_PATH" ]; then
    echo "Error: Video file or directory not found: $VIDEO_PATH"
    echo "Supported: .mp4/.avi/.mov video files or directory containing images"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Generating 4D animation from existing mesh and video"
echo "Mesh file: $DATA_DIR"
echo "Video path: $VIDEO_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Change to project root directory
cd "$PROJECT_ROOT"

# Run inference
python scripts/inference_with_video_mesh.py \
    --config=configs/dyscene.yaml \
    training.resume_ckpt="$PROJECT_ROOT/experiments/checkpoints/ckpt_0000000000060000.pt" \
    model.class_name=model.Pcd_motion.Motion_Latent_Model \
    training.num_shape_samples=16384 \
    training.frames=256 \
    start_frame=0 \
    use_segmentation=True \
    data_dir="$DATA_DIR" \
    video_path="$VIDEO_PATH" \
    output_dir="$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Processing complete!"
echo "Animation output: $OUTPUT_DIR"
echo "=========================================="

