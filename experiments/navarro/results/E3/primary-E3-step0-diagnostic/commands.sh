#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments profile --experiment E3 --checkpoint experiments/navarro/artifacts/primary/initial-s17 --run-id primary-E3-step0-diagnostic
