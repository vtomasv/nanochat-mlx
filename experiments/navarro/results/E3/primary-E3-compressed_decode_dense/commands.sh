#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments inference --config configs/navarro/primary.json --checkpoint /Users/tomasvera/dev/nanochat-mlx/experiments/navarro/artifacts/primary/checkpoints/primary-native-reference-s17/step-1000 --arm compressed_decode_dense --run-id primary-E3-compressed_decode_dense
