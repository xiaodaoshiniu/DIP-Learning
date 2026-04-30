#!/bin/bash
set -e

python bundle_adjustment.py \
  --data-dir data \
  --output-dir results/debug_small \
  --device cpu \
  --max-views 5 \
  --max-points 500 \
  --iters 5 \
  --batch-size 4096 \
  --log-every 1
