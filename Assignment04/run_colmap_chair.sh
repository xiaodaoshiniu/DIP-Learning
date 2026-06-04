#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/dip_hw4_3dgs_project

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

conda activate pytorch_h100
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export PATH="/home/czr/stage2_3dgs_indoor/tools:$PATH"
export COLMAP_BIN="/home/czr/stage2_3dgs_indoor/tools/colmap"
export COLMAP_USE_GPU="${COLMAP_USE_GPU:-0}"
export COLMAP_GPU_INDEX="${COLMAP_GPU_INDEX:-0}"
export COLMAP_NUM_THREADS="${COLMAP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p outputs/logs

backup_dir="outputs/backups/chair_$(date +%Y%m%d_%H%M%S)"
for path in data/chair/database.db data/chair/sparse data/chair/projections; do
  if [ -e "$path" ]; then
    mkdir -p "$backup_dir"
    mv "$path" "$backup_dir/"
  fi
done

python -m compileall gaussian_model.py gaussian_renderer.py train.py data_utils.py mvs_with_colmap.py debug_mvs_by_projecting_pts.py
python mvs_with_colmap.py --data_dir data/chair 2>&1 | tee outputs/logs/colmap_chair.log
python debug_mvs_by_projecting_pts.py --data_dir data/chair 2>&1 | tee outputs/logs/projection_debug_chair.log

echo "COLMAP text model:"
find data/chair/sparse/0_text -maxdepth 1 -type f -printf "%f\n" | sort
echo "Projection images:"
find data/chair/projections -maxdepth 1 -type f | wc -l
