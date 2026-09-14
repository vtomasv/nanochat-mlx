"""Standalone scientific plots from recorded CSV only (matplotlib)."""
from pathlib import Path
import csv
import json
import numpy as np

def render(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=Path(results)
    for exp in ('E1','E2'):
        paths=sorted((results/exp).glob('primary-pilot-*/steps.csv'))
        if not paths:continue
        fig,axs=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
        for path in paths:
            with path.open() as f:rows=list(csv.DictReader(f))
            if not rows:continue
            arm=json.loads((path.parent/'manifest.json').read_text())['arm']
            loss=np.array([float(r['loss']) for r in rows]);seconds=np.cumsum([float(r['seconds']) for r in rows]);tokens=np.cumsum([int(r['tokens']) for r in rows])
            memory=np.array([float(r['mlx_active_peak_bytes'])/1024**3 for r in rows])
            axs[0].plot(tokens/1e6,loss,label=arm,lw=1.2)
            axs[1].plot(seconds,loss,label=arm,lw=1.2)
            axs[2].plot([int(r['step']) for r in rows],memory,label=arm,lw=1.2)
        axs[0].set(xlabel='Processed tokens (millions)',ylabel='Training NLL (nats/token)')
        axs[1].set(xlabel='Cumulative update time (seconds)',ylabel='Training NLL (nats/token)')
        axs[2].set(xlabel='Update (zero based)',ylabel='MLX allocator peak so far (GiB)')
        for ax in axs:ax.grid(alpha=.2)
        axs[0].legend(fontsize=8)
        fig.suptitle(f'{exp} — engineering pilot, seed 101, FP32, M3 Max (one process per arm)')
        out=results/exp/'pilot-curves.png';fig.savefig(out,dpi=160);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)
        print(out)

if __name__=='__main__':
    import sys
    render(sys.argv[1] if len(sys.argv)>1 else 'experiments/navarro/results')
