#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments baseline --experiment E1 --config configs/navarro/primary.json --stage pilot --arm native --seed 101 --steps 100 --run-id primary-pilot-native-s101
