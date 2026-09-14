#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments run --experiment E2 --config configs/navarro/smoke.json --stage smoke --arm staged_raw --seed 101 --steps 10 --run-id smoke-smoke-staged_raw-s101
