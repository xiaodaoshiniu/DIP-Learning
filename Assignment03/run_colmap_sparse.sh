#!/bin/bash
# Sparse-only COLMAP pipeline for machines without CUDA dense reconstruction.

set -e

if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if ! command -v colmap >/dev/null 2>&1; then
    echo "ERROR: colmap is not installed or not in PATH."
    echo "Install it with: conda install -c conda-forge colmap"
    exit 127
fi

DATASET_PATH="${1:-data}"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap_sparse_only"

mkdir -p "$COLMAP_PATH/sparse"

echo "=== Step 1: Feature Extraction ==="
colmap feature_extractor \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

echo "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher \
    --database_path "$COLMAP_PATH/database.db"

echo "=== Step 3: Sparse Reconstruction ==="
colmap mapper \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$COLMAP_PATH/sparse"

echo "=== Done ==="
echo "Sparse model: $COLMAP_PATH/sparse/0"
