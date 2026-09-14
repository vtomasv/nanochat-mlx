#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments run --experiment E2 --config configs/navarro/primary.json --stage pilot --arm staged_grammar --seed 101 --steps 100 --run-id primary-pilot-staged_grammar-s101
