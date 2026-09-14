# Navarro experimental runbook

Run from this checkout with the existing `uv.lock`: `uv run --frozen ...`. MLX requires a macOS session with Metal access; the Codex filesystem sandbox cannot access Metal on this machine. No mock is a substitute. Downloads and large artifacts are ignored by git. Checkpoints are never published.

```sh
uv run --frozen python -m scripts.navarro_experiments preflight --config configs/navarro/primary.json
uv run --frozen python -m scripts.navarro_experiments prepare --config configs/navarro/smoke.json
uv run --frozen python -m scripts.navarro_experiments prepare --config configs/navarro/primary.json
uv run --frozen python -m pytest tests/test_navarro_*.py -v
uv run --frozen python -m scripts.navarro_experiments baseline --config configs/navarro/primary.json --stage reference --seed 17 --run-id primary-native-reference-s17
uv run --frozen python -m scripts.navarro_experiments run --experiment E1 --stage pilot --config configs/navarro/primary.json
uv run --frozen python -m scripts.navarro_experiments run --experiment E2 --stage pilot --config configs/navarro/primary.json
uv run --frozen python -m scripts.navarro_experiments run --experiment E3 --stage pilot --config configs/navarro/primary.json
uv run --frozen python -m scripts.navarro_experiments report --results experiments/navarro/results
```

Each explicit `--arm` runs one process. The default experiment command orchestrates all mandatory arms. A `--run-id` is exclusive: existing directories cannot receive appended duplicate measurements. `--resume-run` returns completed results; interrupted runs use the latest hashed checkpoint and a new run linked to the original (same effective configuration, RNG and schedule). Never extend past update 999.

The shared library builds locally using `clang++ -std=c++17 -O3 -ffp-contract=off -dynamiclib`. Its filename includes the C++ source hash. No fast-math. Rebuilds cannot silently load an old binary. The dependency lock was already present when work began and was retained.

Unknown values remain null. Compression screening is not complete-model performance evidence. Confirmation is evidence-gated: neither a small synthetic fixture nor a single engineering process can establish SUCCESS.

To complete the sequential engineering campaign after the E1 pilot:

```sh
uv run --frozen python -m scripts.navarro_experiments campaign --config configs/navarro/primary.json
```

This command checks recorded real-MLX correctness evidence, invokes the original CLI for 100 updates of its 1000-step schedule, compares its parameters and state to the replay harness, profiles trained checkpoints, runs E2 smoke and pilot, runs each training arm for 100 steps with the original dataloader (`pipeline`), validates the forced E3 layer, and measures all three inference arms. The GPU arms never overlap. Commands and logs are retained. A failed child process stops the campaign without discarding its evidence.

A negative preregistered gate suppresses confirmation. It does not authorize changing thresholds, precision, corpus, optimizer or model architecture to rescue a hypothesis.

Additional diagnostics used for the final dossier (each runs in its own process):

```sh
uv run --frozen python -m scripts.navarro_experiments diagnostics --arm external
uv run --frozen python -m scripts.navarro_experiments detailed --arm native
uv run --frozen python -m scripts.navarro_experiments detailed --arm tape_dense
uv run --frozen python -m scripts.navarro_experiments detailed --arm tape_recompute
uv run --frozen python -m scripts.navarro_experiments detailed --arm tape_bitmap
uv run --frozen python -m scripts.navarro_experiments detailed --arm staged_raw
uv run --frozen python -m scripts.navarro_experiments detailed --arm staged_grammar
uv run --frozen python -m pytest tests/ --junitxml=experiments/navarro/results/correctness-junit.xml
uv run --frozen --with matplotlib python -m nanochat_mlx.experiments.plots experiments/navarro/results
uv run --frozen python -m scripts.navarro_experiments report
```

The optional external diagnostic reads the existing depth-4 SFT checkpoint without altering it. Its tokenizer/training provenance is incomplete, so it is supplemental only. Detailed profiles insert explicit barriers and do not contribute to the principal timing ratios. The final Spanish report is `experiments/navarro/results/informe_final.md`.
