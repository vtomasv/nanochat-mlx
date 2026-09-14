#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments profile --experiment E1 --config configs/navarro/primary.json --checkpoint /Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary/checkpoints/primary-native-reference-s17/step-1000 --run-id primary-E1-trained-1000
