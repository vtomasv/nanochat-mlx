"""Plot the completed row-gradient extension from audited derived values."""
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nanochat_mlx.experiments.v2.records import digest, write_json

root = Path('experiments/navarro/v2/20260915')
source = root / 'verification/closeout-r2/derived.json'
audit = json.loads((source.parent / 'audit.json').read_text())
if digest(source) != audit['derived_outputs']['derived.json']:
    raise ValueError('Changed audited figure input')
rows = json.loads(source.read_text())['rows']
out = root / 'paper/combined-figures-r2'
out.mkdir(exist_ok=False)
plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(8, 5))
ax.axvspan(.99, 1.10, color='#eaf2e9', zorder=0)
ax.axvline(1.10, color='#66805d', ls='--', lw=1)
offsets = {'native': (8, -12), 'tape-dense': (8, 8), 'tape-recompute': (-95, -22),
           'rows-dense': (10, 10), 'rows-bitmap': (10, -8), 'bitmap': (-10, 12)}
for row in rows:
    candidate = row['trial'].startswith('rows-')
    ax.scatter(row['time_ratio_vs_native'], row['mlx_peak_gib'], s=65,
               color='#bb5c29' if candidate else '#397a91', marker='D' if candidate else 'o')
    ax.annotate(row['trial'], (row['time_ratio_vs_native'], row['mlx_peak_gib']),
                textcoords='offset points', xytext=offsets[row['trial']], fontsize=9)
ax.text(1.006, 6.44, 'Presupuesto temporal\n≤ 1,10× nativo', color='#506548', fontsize=9)
ax.set(xlim=(.99, 1.225), ylim=(4.52, 7.05), xlabel='Tiempo por update / nativo de esta extensión',
       ylabel='Pico de memoria MLX (GiB)', title='Filas compactas de gradientes: ahorro con costo temporal')
ax.grid(alpha=.14)
fig.text(.5, .02, 'Un proceso por brazo, 100 updates. Sin IC confirmatorios. Ambos candidatos exceden 1,10×.',
         ha='center', fontsize=8)
fig.tight_layout(rect=(0, .055, 1, 1))
for ext in ('svg', 'pdf', 'png'):
    fig.savefig(out / f'combined.{ext}', dpi=180, bbox_inches='tight')
plt.close(fig)
write_json(out / 'manifest.json', {'source_sha256': digest(Path(__file__)),
    'input_sha256': digest(source), 'audit_sha256': digest(source.parent / 'audit.json'),
    'packages': {p: importlib.metadata.version(p) for p in ('matplotlib', 'numpy')},
    'outputs': {p.name: digest(p) for p in out.iterdir() if p.is_file()},
    'interpretation': 'Within-extension descriptive measurements. Shaded region is only the time gate, not all selection conditions.'})
print(out)
