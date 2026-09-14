"""Generate README figures from committed measurements; never run benchmarks.

uv run --frozen --with matplotlib python docs/navarro/generate_readme_figures.py
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'experiments/navarro/results'
OUTPUT = Path(__file__).resolve().parent / 'images'
SOURCES = {}
BLUE, GRAY, PURPLE, ORANGE = '#2364AA', '#637381', '#8064A2', '#C85A17'
COLORS = [BLUE, GRAY, PURPLE, ORANGE]
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'axes.labelcolor': '#263445',
    'text.color': '#263445', 'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})


def read(relative):
    path = RESULTS / relative
    raw = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)['intervention']


def training(exp, arms):
    return [read(f'{exp}/primary-pilot-{arm}-s101/summary.json') for arm in arms]


def bars(ax, labels, values, title, unit, colors, decimals=2):
    yy = np.arange(len(labels))
    ax.barh(yy, values, color=colors, height=.58)
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc='left', pad=16)
    ax.set_xlabel(unit)
    ax.set_xlim(0, max(values) * 1.28)
    ax.set_axisbelow(True)
    ax.grid(axis='x', alpha=.17)
    for y, v in zip(yy, values):
        ax.text(v + max(values) * .025, y, f'{v:.{decimals}f}', va='center', weight='bold')


def memory(ax, labels, data):
    xx = np.arange(len(labels))
    mlx = [d['mlx_active_peak_bytes'] / 1024**3 for d in data]
    rss = [d['rss_peak_bytes'] / 1024**3 for d in data]
    a = ax.bar(xx - .19, mlx, .36, color=BLUE, label='Pico MLX')
    b = ax.bar(xx + .19, rss, .36, color=ORANGE, label='RSS máximo')
    ax.bar_label(a, fmt='%.2f', padding=3, fontsize=9)
    ax.bar_label(b, fmt='%.2f', padding=3, fontsize=9)
    ax.set_xticks(xx, labels, fontsize=9)
    ax.set_ylabel('GiB (1 GiB = 2³⁰ bytes)')
    ax.set_ylim(0, max(mlx + rss) * 1.24)
    ax.set_title('Memoria: dos medidas separadas', loc='left', pad=16)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.17)


def save(fig, name, title, subtitle, footer):
    fig.suptitle(title, x=.035, ha='left', y=.98, fontsize=19, weight='bold')
    fig.text(.035, .915, subtitle, fontsize=11)
    fig.text(.035, .028, footer, fontsize=10, color='#536477')
    fig.subplots_adjust(left=.14, right=.965, top=.75, bottom=.23, wspace=.72)
    fig.savefig(OUTPUT / f'{name}.png', dpi=175, bbox_inches='tight', pad_inches=.2)
    svg = OUTPUT / f'{name}.svg'
    fig.savefig(svg, bbox_inches='tight', pad_inches=.2)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    e1 = training('E1', ['native', 'tape_dense', 'tape_recompute', 'tape_bitmap'])
    e2 = training('E2', ['native', 'staged_raw', 'staged_grammar'])
    e3 = [read(f'E3/primary-E3-{arm}/summary.json') for arm in
          ['dense_native', 'compressed_decode_dense', 'grammar_direct_cpu']]
    common = 'Apple M3 Max · FP32 · piloto de ingeniería · un proceso por brazo · sin intervalos confirmatorios'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    labels = ['E1 · activaciones', 'E2 · optimizador', 'E3 · inferencia']
    metrics = [
        [e1[-1]['mean_step_seconds']/e1[0]['mean_step_seconds'],
         e2[-1]['mean_step_seconds']/e2[0]['mean_step_seconds'],
         e3[-1]['mean_decode_seconds']/e3[0]['mean_decode_seconds']],
        [ds[-1]['mlx_active_peak_bytes']/ds[0]['mlx_active_peak_bytes'] for ds in (e1,e2,e3)],
        [ds[-1]['rss_peak_bytes']/ds[0]['rss_peak_bytes'] for ds in (e1,e2,e3)],
    ]
    for ax, vals, title in zip(axes, metrics, ['Tiempo relativo', 'Pico MLX relativo', 'RSS relativo']):
        bars(ax, labels, vals, title, 'Intervención / nativo · menor es mejor', ORANGE)
        ax.axvline(1, color=BLUE, ls='--', lw=1.3)
    save(fig, 'overview', 'Tres intervenciones: resultado del sistema completo',
         'La línea azul marca el nativo (1×). E1/E2: tiempo por actualización; E3: tiempo por token de decode.',
         common + '\nMLX y RSS no se suman. E3 usa conversión forzada de una capa: ninguna cumplió el umbral de selección.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    bars(axes[0], ['Nativo', 'Cinta densa', 'Recomputación', 'Bitmap'],
         [d['mean_step_seconds'] for d in e1], 'Costo del paso completo', 'Segundos / actualización', COLORS)
    memory(axes[1], ['Nativo', 'Cinta\ndensa', 'Recomp.', 'Bitmap'], e1)
    ratios = [read(f'E1/primary-E1-trained-{step}/summary.json')['activation_resident_ratio'] for step in (100,500,1000)]
    axes[2].plot([100,500,1000], np.array(ratios)*100, color=ORANGE, marker='o', lw=2.5)
    axes[2].axhline(100, ls='--', color=BLUE, lw=1.3)
    for x, y in zip((100,500,1000), ratios):
        axes[2].annotate(f'{100*y:.1f}%', (x,100*y), xytext=(0,10), textcoords='offset points', ha='center', weight='bold')
    axes[2].set(title='Activaciones P almacenadas', xlabel='Paso de la referencia nativa · semilla 17',
                ylabel='% del tamaño denso (con contenedores)', ylim=(0,115), xlim=(0,1100), xticks=[100,500,1000])
    axes[2].grid(alpha=.17)
    save(fig, 'e1-activations', 'E1 · El bitmap comprime P, pero encarece el entrenamiento',
         'Izquierda y centro: 100 pasos, semilla 101. Derecha: perfiles separados de checkpoints entrenados.',
         common + '\nLa recomputación alcanza el mismo pico MLX que bitmap en este piloto, con menor tiempo y RSS.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    bars(axes[0], ['Nativo', 'Staging RAW', 'Staging + gramática'],
         [d['mean_step_seconds'] for d in e2], 'Costo del paso completo', 'Segundos / actualización', [BLUE,GRAY,ORANGE])
    memory(axes[1], ['Nativo', 'Staging\nRAW', 'Staging +\ngramática'], e2)
    state = [read(f'E2/primary-E2-screen-{step}/summary.json')['eligible_resident_ratio'] for step in (100,500,1000)]
    change = 100*(np.array(state)-1)
    axes[2].scatter([100,500,1000], change, s=75, color=ORANGE, zorder=3)
    axes[2].axhline(0, color=GRAY, lw=1)
    axes[2].axhline(-10, color=BLUE, ls='--', lw=1.3)
    axes[2].text(550, -9.4, 'Umbral previo: −10%', color=BLUE, ha='center', fontsize=10)
    for x, y in zip((100,500,1000),change):
        axes[2].annotate(f'+{y:.3f}%', (x,y), xytext=(0,12), textcoords='offset points', ha='center', fontsize=10)
    axes[2].set(title='Tamaño de estados elegibles', xlabel='Paso de la referencia nativa · semilla 17',
                ylabel='Cambio frente al tamaño denso (%)', ylim=(-12,3), xlim=(0,1100), xticks=[100,500,1000])
    axes[2].grid(alpha=.17)
    save(fig, 'e2-optimizer', 'E2 · Construir la gramática cuesta; los estados terminan en RAW',
         'El control RAW separa el traslado a CPU del intento de compresión. Se conserva el optimizador original.',
         common + '\nLa ausencia de ahorro se midió en los pasos 100, 500 y 1000. Construcción y copias cuentan en el tiempo total.')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    labels = ['Nativo MLX', 'Reconstruir + MLX', 'Producto directo CPU']
    bars(axes[0], labels, [d['mean_decode_seconds']*1000 for d in e3],
         'Decode con continuación fija', 'Milisegundos / token', [BLUE,GRAY,ORANGE])
    bars(axes[1], labels, [d['mean_ttft_seconds']*1000 for d in e3],
         'Tiempo al primer resultado', 'Milisegundos (TTFT desde IDs)', [BLUE,GRAY,ORANGE], decimals=1)
    memory(axes[2], ['Nativo', 'Reconstruir\n+ MLX', 'Directo\nCPU'], e3)
    save(fig, 'e3-inference', 'E3 · Operador correcto en el checkpoint principal; inferencia más lenta',
         'Paso 1000, semilla 17 · 8 prompts de 512 tokens · 128 pasos de decode por prompt · batch 1.',
         common + '\nSe fuerza c_fc del bloque central. La validación externa SFT falla tolerancias y se informa aparte.')

    (OUTPUT/'provenance.json').write_text(json.dumps({
        'generator': str(Path(__file__).relative_to(ROOT)),
        'source_sha256': SOURCES,
        'overview_ratios': dict(zip(('time','mlx_peak','rss_peak'),metrics)),
        'note': 'Figures derived from saved measurements only; no new benchmark or confidence interval.',
    }, indent=2, ensure_ascii=False)+'\n')
    for p in sorted(OUTPUT.glob('*.png')):
        print(p.relative_to(ROOT))


if __name__ == '__main__':
    main()
