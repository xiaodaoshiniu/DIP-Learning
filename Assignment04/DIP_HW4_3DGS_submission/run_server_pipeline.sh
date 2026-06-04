#!/usr/bin/env bash
set -euo pipefail

cd /home/czr/dip_hw4_3dgs_run

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh" >&2
  exit 1
fi

conda activate pytorch_h100
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "python=$(which python)"
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -c "import cv2, numpy, tqdm; print('basic imports ok')"
colmap -h >/dev/null
echo "colmap ok: $(which colmap)"

echo "assignment files:"
find . -maxdepth 2 -type f | sort | sed 's#^\./##' | head -n 40
