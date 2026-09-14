#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments baseline --config configs/navarro/primary.json --stage reference --seed 17 --run-id primary-native-reference-s17
