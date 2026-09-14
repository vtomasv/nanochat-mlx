#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments baseline --config configs/navarro/primary.json --steps 10 --seed 101 --run-id native-layout-check
