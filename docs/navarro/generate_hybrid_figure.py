"""Render only recorded hybrid timings; no generated or estimated data.

uv run --frozen --with matplotlib python docs/navarro/generate_hybrid_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
record = json.loads((ROOT / 'experiments/navarro/verification/hybrid-repair-v2/comparison.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout='constrained')
panels = [
    ('Entrenamiento: actualización completa', 'Segundos por actualización · menor es mejor',
     record['training_mean_step_seconds'], ['baseline', 'hybrid', 'raw', 'native'], 1),
    ('Inferencia: decode con una capa forzada', 'Milisegundos por token · menor es mejor',
     record['inference_mean_decode_seconds'], ['baseline', 'hybrid', 'native'], 1000),
]
labels = {'baseline': 'Codec anterior', 'hybrid': 'Codec híbrido', 'raw': 'Control staged_raw', 'native': 'MLX nativo'}
colors = {'baseline': '#b27039', 'hybrid': '#167c80', 'raw': '#86969d', 'native': '#25486c'}
for ax, (title, xlabel, values, arms, scale) in zip(axes, panels):
    bars = ax.barh([labels[a] for a in arms], [values[a]*scale for a in arms],
                   color=[colors[a] for a in arms], height=.6)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt='%.2f', padding=6, fontsize=11)
    ax.set_xlim(0, max(values[a]*scale for a in arms)*1.22)
    ax.set_title(title, loc='left', pad=15, weight='bold', fontsize=12)
    ax.set_xlabel(xlabel, labelpad=12)
    ax.xaxis.grid(True, alpha=.16)
    ax.set_axisbelow(True)
fig.suptitle('La mezcla reduce el costo del codec; la comparación con MLX sigue siendo necesaria',
             fontsize=13, weight='bold')
fig.supxlabel('M3 Max · FP32 · entrenamiento: 10 pasos por brazo desde checkpoint 100, procesos separados\n'
              'Inferencia: 8 prompts × 128 tokens, brazos alternados en un proceso · mediciones exploratorias sin IC',
              fontsize=9)
out = ROOT / 'docs/navarro/images/hybrid-repair.png'
fig.savefig(out, dpi=170)
fig.savefig(out.with_suffix('.svg'))
svg = out.with_suffix('.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
print(out)
