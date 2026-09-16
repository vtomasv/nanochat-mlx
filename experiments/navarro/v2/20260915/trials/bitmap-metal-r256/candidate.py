"""Editable candidate surface for the external autoresearch agent.

The evaluator, data split, objective and controls live outside this file. Every
trial archives this source before importing it. Implementations must preserve P.
"""
from nanochat_mlx.experiments.v2.bitmap_metal import PackedPositiveGPU


def pack_positive(p, sample_bits=256, strategy='rank'):
    return PackedPositiveGPU.pack(p, sample_bits=sample_bits, strategy=strategy)
