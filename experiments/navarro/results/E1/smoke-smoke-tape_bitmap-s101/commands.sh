#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments run --experiment E1 --config configs/navarro/smoke.json --stage smoke --arm tape_bitmap --seed 101 --steps 10 --run-id smoke-smoke-tape_bitmap-s101
