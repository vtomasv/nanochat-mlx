#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments baseline --experiment E2 --config configs/navarro/smoke.json --stage smoke --arm native --seed 101 --steps 10 --run-id smoke-smoke-native-s101
