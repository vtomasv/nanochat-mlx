#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments repo-native --config configs/navarro/primary.json --steps 10 --run-id repeat
