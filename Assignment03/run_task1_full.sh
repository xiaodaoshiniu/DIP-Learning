#!/bin/bash
set -e

python bundle_adjustment.py \
  --data-dir data \
  --output-dir results/task1_ba_full \
  --device cuda \
  --iters 3000 \
  --batch-size 262144 \
  --log-every 50
