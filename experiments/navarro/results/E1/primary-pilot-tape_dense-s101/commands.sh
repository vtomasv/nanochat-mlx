#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments run --experiment E1 --config configs/navarro/primary.json --stage pilot --arm tape_dense --seed 101 --steps 100 --run-id primary-pilot-tape_dense-s101
