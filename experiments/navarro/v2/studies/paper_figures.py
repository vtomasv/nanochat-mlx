"""Reproducible descriptive figures; no confidence bars from correlated steps."""
import importlib.metadata
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nanochat_mlx.experiments.v2.records import digest,write_json
from nanochat_mlx.experiments.v2.reporting import verified_trials

root=Path('experiments/navarro/v2/20260915');out=root/'paper/figures';out.mkdir(parents=True,exist_ok=False)
trials=verified_trials(root);rows=[r for r in trials if r['trial_id']!='reference']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
labels=[r['trial_id'] for r in rows];y=np.arange(len(rows));colors=['#66717e' if r['scenario']['arm']!='bitmap_gpu' else '#147d92' for r in rows]
fig,axes=plt.subplots(1,2,figsize=(10,5.1),sharey=True)
for ax,key,title in zip(axes,['mean_step_seconds','mlx_active_peak_bytes'],['Tiempo por actualización (s)','Pico de memoria MLX (GiB)']):
 values=[r['metrics'][key]/(1024**3 if key.endswith('bytes') else 1) for r in rows]
 ax.barh(y,values,color=colors,height=.65);ax.set_xlabel(title);ax.set_xlim(0,max(values)*1.21);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
 for pos,value in zip(y,values):ax.text(value+max(values)*.02,pos,f'{value:.3f}',va='center',fontsize=8)
axes[0].set_yticks(y,labels);axes[0].invert_yaxis()
fig.suptitle('nanochat-mlx FP32: ventanas 100–199 desde el mismo checkpoint',fontsize=11)
fig.text(.5,.015,'Un proceso por ensayo, 100 actualizaciones. Descriptivo; sin intervalos confirmatorios.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.045,1,.96))
for ext in ('svg','pdf','png'):fig.savefig(out/f'training.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

census=json.loads((root/'screening/summary.json').read_text())['checkpoint_census']
ops=[json.loads(s) for s in (root/'mechanisms/operators.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for block,color in [(32,'#147d92'),(128,'#bc5f27'),(512,'#66717e')]:
 data=[r for r in census if r['file']=='optimizer.safetensors' and r['rows_per_block']==block]
 axes[0].plot([r['step'] for r in data],[100*r['maximum_savings_under_policy'] for r in data],'-o',label=f'{block} filas',color=color,ms=4)
axes[0].set(xlabel='Actualización nativa',ylabel='Cota superior de ahorro del estado (%)',ylim=(-2,75));axes[0].legend(frameon=False);axes[0].grid(alpha=.15)
for offset,name,label,color in [(-.12,'csr_resident','CSR residente','#147d92'),(.12,'csr_with_build','CSR con construcción','#bc5f27')]:
 ratios=np.array([r['mean_seconds'][name]/r['mean_seconds']['dense'] for r in ops]);x=np.arange(len(ops))
 axes[1].scatter(x+offset,ratios,s=14,label=label,color=color,alpha=.8)
axes[1].axhline(1,color='#444444',linestyle='--',linewidth=1)
axes[1].set(xlabel='Caso H (4 documentos × 8 capas)',ylabel='Tiempo / denso (forward + derivadas)',ylim=(0,18));axes[1].legend(frameon=False);axes[1].grid(alpha=.15)
axes[0].set_title('Cota de diccionario local, no ahorro observado',fontsize=10)
axes[1].set_title('Operador diagnóstico; 28/32 casos fallan tolerancia',fontsize=10)
fig.text(.5,.01,'Microtiempos: un proceso, cinco repeticiones por caso. Incluyen casos incorrectos; no justifican integración.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1))
for ext in ('svg','pdf','png'):fig.savefig(out/f'mechanisms.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
write_json(out/'manifest.json',{'source_sha256':digest(Path(__file__)),'packages':{p:importlib.metadata.version(p) for p in ('matplotlib','numpy')},
    'input_ledger_sha256':digest(root/'ledger.jsonl'),'input_operators_sha256':digest(root/'mechanisms/operators.jsonl'),
    'input_census_sha256':digest(root/'screening/summary.json'),'outputs':{p.name:digest(p) for p in out.iterdir() if p.is_file()},
    'interpretation':'Descriptive figures; no independent-process error bars fabricated.'})
print(out)
