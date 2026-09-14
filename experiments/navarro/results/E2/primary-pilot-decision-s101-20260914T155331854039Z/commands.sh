#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments conclude --experiment E2 --config configs/navarro/primary.json
