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

mkdir -p outputs/logs outputs/backups
out_dir="outputs/chair_simplified_30"
if [ -e "$out_dir" ]; then
  mv "$out_dir" "outputs/backups/chair_simplified_30_$(date +%Y%m%d_%H%M%S)"
fi

python -m compileall gaussian_model.py gaussian_renderer.py train.py data_utils.py evaluate_simplified.py render_3dgs_mv.py

/usr/bin/time -v -o outputs/logs/time_simplified_30.txt \
  python train.py \
    --colmap_dir data/chair \
    --checkpoint_dir "$out_dir" \
    --num_epochs 30 \
    --debug_every 5 \
    --debug_samples 4 \
    --device cuda \
  > outputs/logs/train_simplified_30.log 2>&1

python evaluate_simplified.py \
  --colmap_dir data/chair \
  --checkpoint "$out_dir/checkpoint_000030.pt" \
  --output_dir "$out_dir/eval" \
  --device cuda \
  --save_examples 8 \
  > outputs/logs/eval_simplified_30.log 2>&1

python render_3dgs_mv.py \
  --colmap_dir data/chair \
  --checkpoint "$out_dir/checkpoint_000030.pt" \
  --output "$out_dir/render_mv.mp4" \
  --num_frames 120 \
  --fps 30 \
  --device cuda \
  > outputs/logs/render_mv_simplified_30.log 2>&1

echo "simplified 30 epoch run complete"
