#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments profile --experiment E2 --config configs/navarro/primary.json --checkpoint /Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary/checkpoints/primary-native-reference-s17/step-0500 --run-id primary-E2-screen-500
