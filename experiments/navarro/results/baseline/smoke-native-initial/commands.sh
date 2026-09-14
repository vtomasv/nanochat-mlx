#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments baseline --config configs/navarro/smoke.json --stage smoke --steps 10 --run-id smoke-native-initial
