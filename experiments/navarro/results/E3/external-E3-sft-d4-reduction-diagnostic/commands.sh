#!/bin/sh
cd /Users/tomasvera/dev/nanochat-mlx
uv run --frozen python -m scripts.navarro_experiments diagnostics --arm external --run-id external-E3-sft-d4-reduction-diagnostic
