# Implementation and audit notes

## Baseline and data

The native model and optimizer files are unchanged. `runner.build` mirrors the training script's model construction, parameter groups, muP scaling, batch LR scaling and weight-decay scaling. Checkpoint k denotes k completed updates; every update uses the original 1000-step schedule. Native CLI prefixes use the original entry point and preserve that schedule, in a private base directory with read-only-use data/tokenizer symlinks. A checkpoint hook ends the prefix after weights and optimizer are saved.

The existing corpus has two local shards with 53,248 documents each. SHA-256 exact-document intersection found 43 common documents. The ordinary best-fit dataloader is preserved. Tape includes exact int32 inputs/targets for 1100 updates, and separate validation sequences 0–31 calibration / 32–127 reserved. The packing algorithm loses source document boundaries: document-clustered inference confidence intervals are therefore not available from this tape. This is explicitly recorded, and no such CI is claimed. Actual token counts are processed tokens, not unique corpus tokens.

## E1

`PackedPositive` owns its payload. Bitmap words are uint32, sampled exclusive rank counters uint64. Values are copied into a k-element allocation. The temporary boolean mask and prefix construction are not retained. RAW selection includes host container overhead. Z is never reconstructed; only P=max(Z,0) is stored. Negative/NaN/Inf input is rejected and zero is canonicalized. Construction and unpack are vectorized NumPy operations; host transfers and synchronization are part of full update timing. The random-access rank method uses the N1 sampled count and bounded word popcounts.

`activation_tape.loss_and_grad` is imperative and is not wrapped in global value_and_grad. It materializes and detaches each residual. The local final VJP, explicit MLP backward, local attention VJP and embedding VJP include all trainable inputs, including both residual scales and value embeddings. Attention weights are passed as a flat list because MLX 0.32.2 vjp accepts array lists, not pytrees. Modules are restored after local tracing. Dense and recompute controls use identical scheduling. The compact representation resides on CPU; global RSS must accompany Metal accounting.

## E2 / E3 codec

`native.cpp` constructs CSRV symbols 1+value_id*columns+column, zero delimiter. Exact FP32 bits identify dictionary values. Rules never include delimiters. A deterministic maximum-frequency pair is substituted until none repeats. Frequency ties use ascending pair order. The implementation is compiled but uses repeated frequency scans, so no claim of the authors' linear construction complexity is made. This implementation choice can dominate encode time and limits conclusions about practical algorithm potential.

Rules and final streams use physically packed fixed-width integers for re_iv; re_32 uses 32-bit entries. Rule IDs start after the terminal domain and overflow falls back to RAW. Each block has a 64-byte versioned header, dictionary and aligned streams. Special IEEE values and negative zero use RAW. The reader validates sizes, DAG order, terminal bounds, row order and expansion. Serialization hashes the original bytes. Rows are independent groups of 128. No quantization, column permutation, entropy coder or alternative compressor is present.

`StagedOptimizer` inherits the exact `_muon_step` and `_adamw_step`, including the repository's Muon look-ahead expression. Gradients are materialized before any weight update. Each state's old handle and dense transient are released after update. Both staged arms have the same CPU copy boundaries, and `state` never expands handles. Checkpoint export streams one dense state at a time into standard safetensors; model weights are never overwritten outside experiment directories.

`GrammarLinear` freezes its replaced weight and keeps no permanent dense copy. The compiled right product evaluates rules in topological order, then row symbols. Transpose accumulates row adjoints and propagates in reverse order. re_iv is read directly while computing; no permanently expanded symbol array is hidden in accounting. Prefill runs tiles of 16 vectors. CPU transfers are included. No direct Metal kernel is implemented; GPU acceleration cannot be claimed.

## Measurement and scientific limits

Every arm is a fresh process. MLX arrays, parameters and optimizer states are evaluated before the timer stops, followed by synchronization. Cache limit is 1 GiB; working memory cap is 80 GiB. Warmup occurs in disposable state, followed by checkpoint/RNG reload. RSS is sampled over concurrent process descendants, independently from lifetime resource RSS and Metal peak. A sampler records swap and aborts after three samples with >1 GiB growth. The first reference predates this sampler extension: external readings were zero before and after, not a continuous swap trace.

Engineering curves and ratios do not substitute for five paired processes per 400/900 window or three complete seed trajectories. The decision code can report a preregistered negative screening gate. It never promotes a single pilot to SUCCESS. Changes to acceptance thresholds are not authorized or implemented.

## Native reproducibility diagnostic

The seed101 initial weights and optimizer configuration were identical between native CLI and harness. After ten steps, both native/native and native/harness showed parameter differences. On the checked reserved 1024-token prefix, native/native maximum logit difference was 2.623e-6, native/harness 2.861e-6; both passed the original logit tolerance and had zero measured NLL delta. Native/native maximum individual weight difference (7.65e-4) was comparable to native/harness (6.79e-4). Restoring the original strided token views did not remove the variation. These are measured observations, not a claim of a specific undocumented MLX kernel cause. See `native-variability.json` and the original CLI logs. Bit-identical long training is not claimed. Main replay remains contiguous as initially measured; the original-view run is a separate diagnostic.

One E2 smoke attempt reached ten updates but could not save because an E1 checkpoint shared its old name. The failed run and log were retained with a `failed-checkpoint-collision` suffix. Subsequent checkpoint paths include the experiment ID, preventing cross-experiment collisions.

After smoke, the codec writer was optimized to write packed integers with word operations instead of a bit loop. Adaptive mode computes the exact aligned sizes of both streams and constructs only the selected representation; it still builds the dictionary and full RePair grammar before deciding RAW. Neither symbol selection nor fallback thresholds changed. Both optimizations passed all 23 codec fixtures, recorded in separate JUnit files. The early smoke timing belongs to its recorded source hash and is not mixed with the optimized primary measurements.

Experimental staged checkpoints now persist the selected RAW/re_32/re_iv handles and scalar counters in a versioned `handles/` store. Restore validates every block and file hash without retaining a dense state copy. Earlier engineering checkpoints used dense export and deterministic re-encoding on restore; their bytes and format remain supported. `StagedOptimizer.export_dense(path)` remains the explicit parameter-at-a-time export into the original safetensors format. Neither checkpoint format changes update arithmetic or timing within the measured update.
